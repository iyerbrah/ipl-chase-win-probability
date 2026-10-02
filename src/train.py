"""Train the three win-probability models on the same features and the same split.

Split is by season, so no match (and no era) is shared between train, validation and test:
    train       seasons up to 2023
    validation  2024   (used to tune and to decide when to stop training)
    test        2025-2026   (never touched here; see evaluate.py)

Each model is tuned on the validation season, then refit on train + validation with
those settings, so the final models have seen the most recent pre-test season.

Run:  python src/train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, MONOTONE

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

TRAIN_UNTIL = 2023
VAL_SEASON = 2024
TEST_SEASONS = [2025, 2026]
LABEL = "chase_won"
SEED = 42


def split(df):
    train = df[df.season <= TRAIN_UNTIL]
    val = df[df.season == VAL_SEASON]
    test = df[df.season.isin(TEST_SEASONS)]
    return train, val, test


def fit_logistic(data):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(data[FEATURES], data[LABEL])


def train_logistic(train, val):
    val_loss = log_loss(val[LABEL], fit_logistic(train).predict_proba(val[FEATURES])[:, 1])
    return fit_logistic(pd.concat([train, val])), {}, val_loss


def fit_lightgbm(data, n_trees, num_leaves, min_child_samples, val=None):
    model = lgb.LGBMClassifier(
        n_estimators=n_trees, learning_rate=0.03,
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        monotone_constraints=MONOTONE,
        random_state=SEED, verbose=-1,
    )
    if val is None:
        return model.fit(data[FEATURES], data[LABEL])
    return model.fit(data[FEATURES], data[LABEL],
                     eval_X=val[FEATURES], eval_y=val[LABEL], eval_metric="binary_logloss",
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def train_lightgbm(train, val):
    """Small grid over tree size; each candidate stops when validation log loss stops improving."""
    best_loss, best_params = float("inf"), None
    for num_leaves in (7, 15, 31):
        for min_child_samples in (100, 400):
            model = fit_lightgbm(train, 2000, num_leaves, min_child_samples, val=val)
            loss = model.best_score_["valid_0"]["binary_logloss"]
            if loss < best_loss:
                best_loss = loss
                best_params = {"n_trees": model.best_iteration_, "num_leaves": num_leaves,
                               "min_child_samples": min_child_samples}
    return fit_lightgbm(pd.concat([train, val]), **best_params), best_params, best_loss


def new_mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-4,
                         batch_size=512, max_iter=1, warm_start=True, random_state=SEED)


def train_mlp(train, val, max_epochs=60, patience=8):
    """Two hidden layers; trained one epoch at a time, with the epoch count chosen on validation."""
    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_val = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES])
    net, best_loss, best_epoch = new_mlp(), float("inf"), 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            net.fit(X_train, train[LABEL])
            loss = log_loss(val[LABEL], net.predict_proba(X_val)[:, 1])
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
            elif epoch - best_epoch >= patience:
                break

        both = pd.concat([train, val])
        scaler = StandardScaler().fit(both[FEATURES])
        net = new_mlp()
        for _ in range(best_epoch):
            net.fit(scaler.transform(both[FEATURES]), both[LABEL])
    return make_pipeline(scaler, net), {"hidden_layers": [64, 32], "epochs": best_epoch}, best_loss


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "chases.parquet")
    train, val, test = split(df)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        print(f"{name:11s} {part.match_id.nunique():5d} matches  {len(part):7,d} deliveries")

    MODELS.mkdir(exist_ok=True)
    trainers = {"logistic": train_logistic, "lightgbm": train_lightgbm, "mlp": train_mlp}
    settings = {}
    for name, trainer in trainers.items():
        model, params, val_loss = trainer(train, val)
        joblib.dump(model, MODELS / f"{name}.joblib")
        settings[name] = {**params, "validation_log_loss": round(val_loss, 4)}
        print(f"{name:9s} validation log loss {val_loss:.4f}  {params}")

    meta = {
        "features": FEATURES,
        "train_until": TRAIN_UNTIL, "val_season": VAL_SEASON, "test_seasons": TEST_SEASONS,
        "train_chase_win_rate": float(train.groupby("match_id")[LABEL].first().mean()),
        "models": settings,
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
