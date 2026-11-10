"""Every number the README quotes comes from the committed reports, so the two can't drift apart."""
import json

import pandas as pd

README = open("README.md", encoding="utf-8").read()
METRICS = json.load(open("reports/metrics.json"))
LEVERAGE = json.load(open("models/leverage.json"))
CHASES = pd.read_parquet("data/processed/chases.parquet")


def test_results_table():
    for name in ["naive", "logistic", "lightgbm", "mlp"]:
        m = METRICS["overall"][name]
        assert f"{m['accuracy']:.1%}" in README, name
        assert f"{m['log_loss']:.3f}" in README, name
        assert f"{m['brier']:.3f}" in README, name


def test_test_set_size():
    assert f"{METRICS['test_matches']} matches ({METRICS['test_deliveries']:,} deliveries)" in README


def test_dataset_size():
    assert f"{CHASES.match_id.nunique():,} matches, {len(CHASES):,} deliveries" in README


def test_leverage_concentration():
    assert f"{LEVERAGE['test_over_breaks']:,} over-breaks" in README
    assert f"top 10% of over-breaks hold {LEVERAGE['top_10pct_share']:.0%}" in README
    assert f"top 20% hold {LEVERAGE['top_20pct_share']:.0%}" in README
