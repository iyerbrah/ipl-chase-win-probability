"""Shared feature definitions, used by the dataset builder, training, leverage and the app."""
import numpy as np
import pandas as pd

FEATURES = [
    "runs_needed",
    "balls_remaining",
    "wickets_in_hand",
    "required_run_rate",
    "current_run_rate",
    "target",
    "venue_avg_score",
    "league_avg_score",
]

# LightGBM monotonic constraints, in FEATURES order:
# more runs needed can never help the chasing side; more balls or wickets can never hurt.
MONOTONE = [-1, 1, 1, 0, 0, 0, 0, 0]

TOTAL_BALLS = 120
MAX_RATE = 36.0  # six sixes an over; caps the rates so late-innings rows don't explode


def make_features(runs_needed, balls_remaining, wickets_in_hand, target,
                  venue_avg_score, league_avg_score):
    """Build the model's feature table from raw match state (scalars or arrays)."""
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_remaining = np.asarray(balls_remaining, dtype=float)
    target = np.asarray(target, dtype=float)
    balls_bowled = TOTAL_BALLS - balls_remaining
    score = target - runs_needed

    rrr = runs_needed * 6 / np.maximum(balls_remaining, 1)
    crr = np.where(balls_bowled > 0, score * 6 / np.maximum(balls_bowled, 1), 0.0)

    return pd.DataFrame({
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "wickets_in_hand": np.asarray(wickets_in_hand, dtype=float),
        "required_run_rate": np.clip(rrr, 0, MAX_RATE),
        "current_run_rate": np.clip(crr, 0, MAX_RATE),
        "target": target,
        "venue_avg_score": np.asarray(venue_avg_score, dtype=float),
        "league_avg_score": np.asarray(league_avg_score, dtype=float),
    })[FEATURES]
