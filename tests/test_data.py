"""The committed dataset: one season split with no leaks, and predictions for every delivery."""
import json

import pandas as pd

META = json.load(open("models/meta.json"))
CHASES = pd.read_parquet("data/processed/chases.parquet")
PREDICTIONS = pd.read_parquet("data/processed/predictions.parquet")


def test_no_match_is_in_more_than_one_split():
    seasons_per_match = CHASES.groupby("match_id").season.nunique()
    assert (seasons_per_match == 1).all()
    test = CHASES[CHASES.season.isin(META["test_seasons"])].match_id
    train = CHASES[CHASES.season <= META["val_season"]].match_id
    assert not set(test) & set(train)


def test_split_is_by_season():
    assert META["train_until"] < META["val_season"] < min(META["test_seasons"])
    test = PREDICTIONS[PREDICTIONS.split == "test"]
    assert set(test.season) == set(META["test_seasons"])


def test_every_delivery_has_a_probability_from_each_model():
    assert len(PREDICTIONS) == len(CHASES)
    for name in ["logistic", "lightgbm", "mlp"]:
        p = PREDICTIONS[f"wp_{name}"]
        assert p.notna().all() and p.between(0, 1).all()


def test_features_match_the_feature_definitions():
    assert META["features"] == ["runs_needed", "balls_remaining", "wickets_in_hand", "required_run_rate",
                                "current_run_rate", "target", "venue_avg_score", "league_avg_score"]
