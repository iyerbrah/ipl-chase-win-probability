"""make_features(): the eight numbers every model sees, built from the raw match state."""
import numpy as np
import pytest

from features import FEATURES, MAX_RATE, MONOTONE, make_features


def state(runs_needed, balls_remaining, wickets=5, target=180):
    n = np.broadcast(runs_needed, balls_remaining).size

    def full(v):
        return np.broadcast_to(v, n)

    return make_features(full(runs_needed), full(balls_remaining), full(wickets), full(target),
                         venue_avg_score=full(170), league_avg_score=full(175))


def test_rates_mid_chase():
    row = state(runs_needed=60, balls_remaining=30).iloc[0]
    assert row.required_run_rate == pytest.approx(12.0)   # 60 runs in 5 overs
    assert row.current_run_rate == pytest.approx(8.0)     # 120 runs in 15 overs


def test_first_ball_has_no_current_rate():
    assert state(runs_needed=180, balls_remaining=120).iloc[0].current_run_rate == 0.0


def test_rates_are_capped():
    row = state(runs_needed=50, balls_remaining=1).iloc[0]
    assert row.required_run_rate == MAX_RATE


def test_columns_and_constraints_line_up():
    features = state(runs_needed=[60, 30], balls_remaining=[30, 12])
    assert list(features.columns) == FEATURES
    assert len(MONOTONE) == len(FEATURES)
    assert np.isfinite(features.to_numpy()).all()
