"""Win probability for any match state, and the leverage built on top of it.

Leverage is not learned. For a given state we ask the model what the win probability
would be after each possible next ball, and take the average absolute change, weighted
by how often each outcome happens in that phase of the innings.

    leverage index = expected swing of this ball / expected swing of an average ball

Run:  python src/leverage.py     (writes predictions for every delivery, plus a summary)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import TOTAL_BALLS, make_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["logistic", "lightgbm", "mlp"]

# outcome -> (runs scored, balls used, wickets lost)
OUTCOMES = {
    "0": (0, 1, 0), "1": (1, 1, 0), "2": (2, 1, 0), "3": (3, 1, 0),
    "4": (4, 1, 0), "6": (6, 1, 0), "wicket": (0, 1, 1), "extra": (1, 0, 0),
}
PHASES = ["powerplay", "middle", "death"]


def phase_of(balls_remaining):
    """Overs 1-6 powerplay, 7-15 middle, 16-20 death."""
    over = (TOTAL_BALLS - np.asarray(balls_remaining)) // 6 + 1
    return np.select([over <= 6, over <= 15], ["powerplay", "middle"], default="death")


def win_probability(model, runs_needed, balls_remaining, wickets_in_hand,
                    target, venue_avg_score, league_avg_score):
    """Model win probability, with finished matches set to their known result."""
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_remaining = np.asarray(balls_remaining, dtype=float)
    wickets_in_hand = np.asarray(wickets_in_hand, dtype=float)

    X = make_features(np.maximum(runs_needed, 0), np.maximum(balls_remaining, 0), wickets_in_hand,
                      target, venue_avg_score, league_avg_score)
    wp = model.predict_proba(X)[:, 1]

    out_of_resources = (balls_remaining <= 0) | (wickets_in_hand <= 0)
    wp = np.where(out_of_resources, np.where(runs_needed == 1, 0.5, 0.0), wp)  # scores level = tie
    return np.where(runs_needed <= 0, 1.0, wp)


def outcome_frequencies(df):
    """Share of each next-ball outcome, by phase of the innings."""
    counts = pd.crosstab(phase_of(df.balls_remaining), df.outcome)
    counts = counts.reindex(index=PHASES, columns=list(OUTCOMES), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def expected_swing(model, df, freqs, wp_now=None):
    """Average absolute change in win probability over the possible next balls."""
    args = (df.target.values, df.venue_avg_score.values, df.league_avg_score.values)
    if wp_now is None:
        wp_now = win_probability(model, df.runs_needed, df.balls_remaining, df.wickets_in_hand, *args)
    weights = freqs.loc[phase_of(df.balls_remaining)]

    swing = np.zeros(len(df))
    for outcome, (runs, balls, wickets) in OUTCOMES.items():
        wp_next = win_probability(model, df.runs_needed.values - runs, df.balls_remaining.values - balls,
                                  df.wickets_in_hand.values - wickets, *args)
        swing += weights[outcome].values * np.abs(wp_next - wp_now)
    return swing


def concentration(values, top_share):
    """Share of the total held by the top `top_share` fraction of values."""
    ordered = np.sort(values)[::-1]
    k = max(1, int(round(top_share * len(ordered))))
    return float(ordered[:k].sum() / ordered.sum())


def main():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    models = {name: joblib.load(ROOT / "models" / f"{name}.joblib") for name in MODEL_NAMES}
    best = min(MODEL_NAMES, key=lambda n: meta["models"][n]["validation_log_loss"])

    df = pd.read_parquet(ROOT / "data" / "processed" / "chases.parquet")
    seen = df.season <= meta["val_season"]          # seasons the models were fit on
    is_test = df.season.isin(meta["test_seasons"])

    args = (df.runs_needed, df.balls_remaining, df.wickets_in_hand,
            df.target, df.venue_avg_score, df.league_avg_score)
    for name, model in models.items():
        df[f"wp_{name}"] = win_probability(model, *args)

    freqs = outcome_frequencies(df[seen])
    df["swing"] = expected_swing(models[best], df, freqs, wp_now=df[f"wp_{best}"].values)
    average_swing = float(df.loc[seen, "swing"].mean())
    df["leverage"] = df["swing"] / average_swing
    df["split"] = np.where(is_test, "test", "train")

    df.to_parquet(ROOT / "data" / "processed" / "predictions.parquet", index=False)

    # An over-break is valued by the leverage of the first ball after it.
    breaks = df[is_test].groupby(["match_id", "over"]).first().reset_index()
    summary = {
        "model": best,
        "average_swing": average_swing,
        "outcome_frequencies": freqs.round(4).to_dict(orient="index"),
        "test_over_breaks": int(len(breaks)),
        "top_10pct_share": concentration(breaks.leverage.values, 0.10),
        "top_20pct_share": concentration(breaks.leverage.values, 0.20),
        "mean_leverage_by_over": breaks.groupby("over").leverage.mean().round(3).to_dict(),
    }
    (ROOT / "models" / "leverage.json").write_text(json.dumps(summary, indent=2))

    print(f"leverage model: {best}   average swing per ball: {average_swing:.4f}")
    print(freqs.round(3).to_string())
    print(f"test over-breaks: {len(breaks):,}")
    print(f"top 10% of over-breaks hold {summary['top_10pct_share']:.1%} of total leverage")
    print(f"top 20% of over-breaks hold {summary['top_20pct_share']:.1%} of total leverage")


if __name__ == "__main__":
    main()
