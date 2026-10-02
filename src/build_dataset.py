"""Turn the Cricsheet IPL match files into one row per delivery of each run chase.

Each row holds the match state *before* the ball is bowled and the label
`chase_won` (1 if the chasing team won the match).

Run:  python src/build_dataset.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import TOTAL_BALLS, make_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ipl_json"
OUT = ROOT / "data" / "processed"

# The same ground appears under several names across seasons.
VENUE_ALIASES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",
}

VENUE_PRIOR_WEIGHT = 5   # a venue's average is pulled toward the league's until it has history
LEAGUE_WINDOW = 100      # matches used for the recent league scoring level
MIN_LEAGUE_HISTORY = 20  # matches needed before the league average is trusted
DEFAULT_SCORE = 160.0    # used only for the very first matches, before enough history exists


def clean_venue(name):
    name = name.split(",")[0].strip()
    return VENUE_ALIASES.get(name, name)


def innings_total(innings):
    return sum(d["runs"]["total"] for over in innings["overs"] for d in over["deliveries"])


def outcome_class(delivery):
    """Collapse a delivery into the outcome classes used by the leverage calculation."""
    extras = delivery.get("extras", {})
    if "wides" in extras or "noballs" in extras:
        return "extra"
    if delivery.get("wickets"):
        return "wicket"
    runs = delivery["runs"]["total"]
    if runs >= 6:
        return "6"
    if runs >= 4:
        return "4"
    return str(runs)


def load_matches():
    matches = []
    for path in glob.glob(str(RAW / "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        info = data["info"]
        matches.append({
            "match_id": int(Path(path).stem),
            "date": info["dates"][0],
            "season": int(info["dates"][0][:4]),
            "venue": clean_venue(info["venue"]),
            "stage": info.get("event", {}).get("stage", "League"),
            "info": info,
            "innings": data["innings"],
        })
    return sorted(matches, key=lambda m: (m["date"], m["match_id"]))


def add_scoring_context(matches):
    """Attach venue and league scoring levels using only matches played earlier."""
    venue_scores, league_scores = {}, []
    for m in matches:
        enough_history = len(league_scores) >= MIN_LEAGUE_HISTORY
        league_avg = np.mean(league_scores[-LEAGUE_WINDOW:]) if enough_history else DEFAULT_SCORE
        past = venue_scores.get(m["venue"], [])
        m["league_avg_score"] = league_avg
        m["venue_avg_score"] = (sum(past) + VENUE_PRIOR_WEIGHT * league_avg) / (len(past) + VENUE_PRIOR_WEIGHT)

        # Only full, uninterrupted first innings update the history.
        first = m["innings"][0] if m["innings"] else None
        full_match = m["info"].get("overs") == 20 and m["info"]["outcome"].get("method") is None
        if first and full_match and m["info"]["outcome"].get("result") != "no result":
            total = innings_total(first)
            venue_scores.setdefault(m["venue"], []).append(total)
            league_scores.append(total)


def skip_reason(m):
    outcome = m["info"]["outcome"]
    if "winner" not in outcome:
        return outcome.get("result", "no winner")
    if outcome.get("method"):
        return "rain-adjusted (D/L)"
    if len(m["innings"]) < 2:
        return "no second innings"
    target = m["innings"][1].get("target", {})
    if target.get("overs", 20) != 20 or m["info"].get("overs") != 20:
        return "shortened match"
    return None


def chase_rows(m):
    chase = m["innings"][1]
    target = chase.get("target", {}).get("runs") or innings_total(m["innings"][0]) + 1
    batting = chase["team"]
    bowling = next(t for t in m["info"]["teams"] if t != batting)
    chase_won = int(m["info"]["outcome"]["winner"] == batting)

    rows, score, wickets, legal_balls = [], 0, 0, 0
    for over in chase["overs"]:
        for d in over["deliveries"]:
            extras = d.get("extras", {})
            is_legal = "wides" not in extras and "noballs" not in extras
            rows.append({
                "match_id": m["match_id"], "date": m["date"], "season": m["season"],
                "venue": m["venue"], "stage": m["stage"],
                "batting_team": batting, "bowling_team": bowling,
                "over": over["over"] + 1,
                "batter": d["batter"], "bowler": d["bowler"],
                # state before the ball
                "runs_needed": target - score,
                "balls_remaining": TOTAL_BALLS - legal_balls,
                "wickets_in_hand": 10 - wickets,
                "target": target,
                "venue_avg_score": m["venue_avg_score"],
                "league_avg_score": m["league_avg_score"],
                # what happened on the ball
                "runs_on_ball": d["runs"]["total"],
                "wicket_on_ball": int(bool(d.get("wickets"))),
                "legal_ball": int(is_legal),
                "outcome": outcome_class(d),
                "chase_won": chase_won,
            })
            score += d["runs"]["total"]
            wickets += len(d.get("wickets", []))
            legal_balls += is_legal
    return rows


def main():
    matches = load_matches()
    add_scoring_context(matches)

    rows, skipped = [], {}
    for m in matches:
        reason = skip_reason(m)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows.extend(chase_rows(m))

    df = pd.DataFrame(rows)
    # Rows after the innings should have ended would be data errors.
    df = df[(df.balls_remaining > 0) & (df.runs_needed > 0) & (df.wickets_in_hand > 0)]
    feats = make_features(df.runs_needed, df.balls_remaining, df.wickets_in_hand,
                          df.target, df.venue_avg_score, df.league_avg_score)
    df["required_run_rate"] = feats["required_run_rate"].values
    df["current_run_rate"] = feats["current_run_rate"].values
    df["ball_no"] = df.groupby("match_id").cumcount() + 1

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "chases.parquet", index=False)

    print(f"matches read:     {len(matches)}")
    print(f"matches skipped:  {sum(skipped.values())}  {skipped}")
    print(f"matches kept:     {df.match_id.nunique()}")
    print(f"deliveries:       {len(df):,}")
    print(f"chase win rate:   {df.groupby('match_id').chase_won.first().mean():.3f}")
    print(df.groupby("season").match_id.nunique().to_string())


if __name__ == "__main__":
    main()
