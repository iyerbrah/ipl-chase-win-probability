"""Score the three models on the held-out test seasons (2025-2026).

Every prediction is compared with the recorded match result. Writes:
    reports/metrics.json      overall and by-phase scores
    reports/calibration.csv   predicted vs actual win rate, in probability bins
    reports/calibration.png   the same as a chart

Run:  python src/evaluate.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import FEATURES
from leverage import MODEL_NAMES, PHASES, phase_of

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABEL = "chase_won"
BINS = np.linspace(0, 1, 11)


def score(y, p):
    constant = np.ptp(p) == 0   # AUC is undefined for a constant prediction
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": 0.5 if constant else roc_auc_score(y, p),
    }


def calibration_table(y, p, model):
    bins = pd.cut(p, BINS, include_lowest=True)
    table = pd.DataFrame({"predicted": p, "actual": y}).groupby(bins, observed=True).agg(
        predicted=("predicted", "mean"), actual=("actual", "mean"), deliveries=("actual", "size"))
    return table.reset_index(drop=True).assign(model=model)


def main():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    df = pd.read_parquet(ROOT / "data" / "processed" / "chases.parquet")
    test = df[df.season.isin(meta["test_seasons"])]
    y = test[LABEL].values
    phases = phase_of(test.balls_remaining)

    predictions = {"naive": np.full(len(test), meta["train_chase_win_rate"])}
    for name in MODEL_NAMES:
        model = joblib.load(ROOT / "models" / f"{name}.joblib")
        predictions[name] = model.predict_proba(test[FEATURES])[:, 1]

    overall = {name: score(y, p) for name, p in predictions.items()}
    by_phase = {name: {ph: score(y[phases == ph], p[phases == ph]) for ph in PHASES}
                for name, p in predictions.items()}
    calibration = pd.concat(calibration_table(y, predictions[n], n) for n in MODEL_NAMES)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps({
        "test_seasons": meta["test_seasons"],
        "test_matches": int(test.match_id.nunique()),
        "test_deliveries": int(len(test)),
        "overall": overall, "by_phase": by_phase,
    }, indent=2))
    calibration.to_csv(REPORTS / "calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for name in MODEL_NAMES:
        part = calibration[calibration.model == name]
        ax.plot(part.predicted, part.actual, marker="o", label=name)
    ax.set(xlabel="Predicted win probability", ylabel="Actual win rate",
           title=f"Calibration on test seasons {meta['test_seasons'][0]}-{meta['test_seasons'][-1]}")
    ax.legend()
    fig.savefig(REPORTS / "calibration.png", dpi=150, bbox_inches="tight")

    print(f"test: {test.match_id.nunique()} matches, {len(test):,} deliveries")
    print(pd.DataFrame(overall).T.round(4).to_string())
    print("\naccuracy by phase")
    print(pd.DataFrame({n: {ph: s["accuracy"] for ph, s in by_phase[n].items()} for n in by_phase}).T.round(4).to_string())
    print("\nlog loss by phase")
    print(pd.DataFrame({n: {ph: s["log_loss"] for ph, s in by_phase[n].items()} for n in by_phase}).T.round(4).to_string())


if __name__ == "__main__":
    main()
