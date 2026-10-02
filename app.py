"""IPL chase win probability: three models compared, and the leverage built on the best one.

Two tabs:
    Win probability   ball-by-ball predictions, model comparison, accuracy
    Leverage          which moments of a match matter most

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODELS = {"logistic": "Logistic regression", "lightgbm": "LightGBM", "mlp": "Neural network"}
NAIVE = "No model (always back the chasing team)"
# Fixed colour per model, the same in every chart.
COLORS = {"logistic": "#2a78d6", "lightgbm": "#eb6834", "mlp": "#1baf7a"}
MUTED = "#898781"

st.set_page_config(page_title="IPL Win Probability", page_icon="🏏", layout="centered")


@st.cache_data
def load_predictions():
    return pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")


@st.cache_data
def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text())


def line_chart(x_title, y_title, height):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=x_title), yaxis=dict(title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def match_chart(y_title, height):
    fig = line_chart("Overs bowled in the chase", y_title, height)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(range=[0, 20], dtick=2)
    return fig


def add_situation_hover(fig, match, y):
    """Invisible trace that puts the match situation at the top of the hover box."""
    fig.add_trace(go.Scatter(x=match.overs_bowled, y=y, mode="lines", line=dict(width=0), showlegend=False,
                             customdata=[situation(row) for row in match.itertuples()],
                             hovertemplate="%{customdata}<extra></extra>"))
    fig.update_xaxes(unifiedhovertitle=dict(text="Match situation"))


def overview(df, metrics):
    seasons = metrics["test_seasons"]
    a, b, c, d = st.columns(4)
    a.metric("Matches", f"{df.match_id.nunique():,}")
    b.metric("Deliveries", f"{len(df):,}")
    c.metric("Models compared", len(MODELS))
    d.metric("Tested on", f"{seasons[0]}-{seasons[-1]}")


def pick_match(df):
    season_col, match_col = st.columns([1, 3])
    season = season_col.selectbox("Season", sorted(df.season.unique(), reverse=True))
    matches = (df[df.season == season].groupby("match_id")
               .agg(date=("date", "first"), batting=("batting_team", "first"),
                    bowling=("bowling_team", "first"), peak=("leverage", "max"))
               .sort_values("date", ascending=False))
    labels = {mid: f"{m.date}  ·  {m.bowling} v {m.batting}" for mid, m in matches.iterrows()}
    tightest = list(labels).index(matches.peak.idxmax())
    match_id = match_col.selectbox("Match", list(labels), index=tightest, format_func=labels.get)

    match = df[df.match_id == match_id].sort_values("ball_no").copy()
    match["overs_bowled"] = (120 - match.balls_remaining) / 6
    first = match.iloc[0]
    st.markdown(f"**{first.batting_team}** needed **{int(first.target)}** to beat {first.bowling_team}, "
                f"and **{'won' if first.chase_won else 'lost'}**.")
    return match


def over_notation(balls_remaining):
    """Balls bowled so far in cricket notation: 17.3 is 17 overs and 3 balls."""
    bowled = 120 - int(balls_remaining)
    return f"{bowled // 6}.{bowled % 6}"


def equation(row):
    balls = "1 ball" if row.balls_remaining == 1 else f"{row.balls_remaining} balls"
    wickets = "1 wicket" if row.wickets_in_hand == 1 else f"{row.wickets_in_hand} wickets"
    return f"{row.runs_needed} needed off {balls}, {wickets} in hand"


def situation(row):
    return f"After {over_notation(row.balls_remaining)} overs: {equation(row)}"


def ball_event(row):
    if row.wicket_on_ball:
        return "Wicket"
    names = {0: "Dot ball", 1: "1 run", 4: "Four", 6: "Six"}
    return names.get(row.runs_on_ball, f"{row.runs_on_ball} runs")


def turning_points(match, model, count=3):
    """The balls that moved the win probability most, kept at least two overs apart."""
    wp = match[f"wp_{model}"]
    wp_after = wp.shift(-1).fillna(float(match.chase_won.iloc[0]))   # the last ball settles the match
    swing = (wp_after - wp) * 100
    picked = []
    for index in swing.abs().sort_values(ascending=False).index:
        if all(abs(match.ball_no[index] - match.ball_no[other]) >= 12 for other in picked):
            picked.append(index)
        if len(picked) == count:
            break
    return match.loc[picked].assign(swing=swing[picked])


def slider_key(match, tab):
    return f"{tab}_ball_{match.match_id.iloc[0]}"


def selected_ball(match, tab):
    """The ball chosen on this tab's slider (drawn below the chart); starts at the tensest ball."""
    default = int(match.ball_no[match.leverage.idxmax()])
    return st.session_state.get(slider_key(match, tab), default)


def win_probability_chart(match, main_model, selected):
    first = match.iloc[0]
    fig = match_chart("Win probability (%)", height=420)
    fig.add_hline(y=50, line_dash="dot", line_color=MUTED, line_width=1)

    add_situation_hover(fig, match, match[f"wp_{main_model}"] * 100)
    for name, label in MODELS.items():
        is_main = name == main_model
        fig.add_trace(go.Scatter(x=match.overs_bowled, y=match[f"wp_{name}"] * 100, mode="lines",
                                 name=label, opacity=1 if is_main else 0.55,
                                 line=dict(color=COLORS[name], width=3 if is_main else 1.5),
                                 hovertemplate="%{y:.0f}%"))
    wickets = match[match.wicket_on_ball == 1]
    fig.add_trace(go.Scatter(x=wickets.overs_bowled, y=wickets[f"wp_{main_model}"] * 100, mode="markers",
                             name="Wicket", marker=dict(color=MUTED, size=9, symbol="x"),
                             hoverinfo="skip"))

    for point in turning_points(match, main_model).itertuples():
        fig.add_annotation(x=point.overs_bowled, y=getattr(point, f"wp_{main_model}") * 100,
                           text=f"{ball_event(point)} {point.swing:+.0f}", showarrow=True,
                           arrowhead=0, arrowcolor=MUTED, ax=0, ay=-34 if point.swing > 0 else 34,
                           xanchor="right" if point.overs_bowled > 16 else "center")

    chosen = match[match.ball_no == selected].iloc[0]
    fig.add_vline(x=chosen.overs_bowled, line_dash="dash", line_color=MUTED, line_width=1)

    # The top of the chart belongs to the chasing team, the bottom to the defending team.
    for y, anchor, text in ((1, "top", f"▲ {first.batting_team} winning"),
                            (0, "bottom", f"▼ {first.bowling_team} winning")):
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=y, yanchor=anchor, xanchor="left",
                           text=text, showarrow=False, font=dict(color=MUTED))
    fig.update_yaxes(range=[-10, 110], tickvals=[0, 25, 50, 75, 100])
    return fig


def ball_slider(match, tab, selected):
    """Slider to pick a ball; shows the score at that moment and returns that ball's row."""
    selected = st.select_slider("Pick a moment in the chase (overs bowled)", options=list(match.ball_no),
                                value=selected, format_func=lambda ball: over_notation(
                                    match.balls_remaining[match.ball_no == ball].iloc[0]),
                                key=slider_key(match, tab))
    chosen = match[match.ball_no == selected].iloc[0]
    score = int(chosen.target - chosen.runs_needed)
    st.markdown(f"**{chosen.batting_team} {score}/{10 - int(chosen.wickets_in_hand)}.** {situation(chosen)}.")
    return chosen


def win_probability_tab(match, metrics, main_model):
    st.markdown("**Definition.** Win probability is the chasing team's chance of winning, predicted before "
                "each ball from runs needed, balls remaining, wickets in hand, run rates, the target "
                "and how high-scoring the ground is.")

    st.subheader("Ball by ball")
    selected = selected_ball(match, "wp")
    st.plotly_chart(win_probability_chart(match, main_model, selected), width="stretch")
    chosen = ball_slider(match, "wp", selected)
    for column, (name, label) in zip(st.columns(len(MODELS)), MODELS.items()):
        column.metric(label, f"{chosen[f'wp_{name}']:.0%}")

    seasons = metrics["test_seasons"]
    overall, by_phase = metrics["overall"], metrics["by_phase"]
    labels = {"naive": NAIVE, **MODELS}

    st.subheader(f"Model comparison on unseen seasons ({seasons[0]}-{seasons[-1]}, {metrics['test_matches']} matches)")
    table = pd.DataFrame(overall).T.rename(index=labels)[["accuracy", "log_loss", "brier"]]
    table.columns = ["Accuracy", "Log loss", "Brier score"]
    st.table(table.style.format({"Accuracy": "{:.1%}", "Log loss": "{:.3f}", "Brier score": "{:.3f}"}))
    st.markdown("**Definitions.**\n"
                "- Accuracy: share of balls where the model's favourite went on to win.\n"
                "- Brier score: average squared gap between the predicted probability and the result (0 or 1).\n"
                "- Log loss: like the Brier score, but confident wrong predictions are penalised more heavily.")

    st.subheader("Accuracy by phase of the chase")
    phases = pd.DataFrame({MODELS[name]: {phase: scores["accuracy"] for phase, scores in by_phase[name].items()}
                           for name in MODELS}).T
    phases.columns = ["Overs 1-6", "Overs 7-15", "Overs 16-20"]
    st.table(phases.style.format("{:.1%}"))

    accuracies = [overall[name]["accuracy"] for name in MODELS]
    st.markdown(
        "**Observations.**\n"
        f"- All three models reach {min(accuracies):.0%}-{max(accuracies):.0%} accuracy, "
        f"against {overall['naive']['accuracy']:.0%} with no model.\n"
        f"- Logistic regression and the neural network are level on log loss "
        f"({overall['logistic']['log_loss']:.3f} and {overall['mlp']['log_loss']:.3f}); "
        f"LightGBM is behind ({overall['lightgbm']['log_loss']:.3f}).\n"
        f"- Accuracy rises through the chase: {by_phase['logistic']['powerplay']['accuracy']:.0%} in overs 1-6 "
        f"to {by_phase['logistic']['death']['accuracy']:.0%} in overs 16-20 for logistic regression.")
    st.info("**Conclusion.** The simplest model performs as well as the most complex one. "
            "Logistic regression is used to calculate leverage.")


def leverage_tab(match, summary):
    st.markdown("**Definitions.**\n"
                "- Leverage: the expected change in win probability from the next ball, "
                "relative to an average ball (1.0 = average).\n"
                "- Over-break value: the leverage at the start of the over that follows the break.")

    st.subheader("Ball by ball")
    fig = match_chart("Leverage (1.0 = average ball)", height=320)
    fig.add_hline(y=1, line_dash="dot", line_color=MUTED, line_width=1)
    add_situation_hover(fig, match, match.leverage)
    fig.add_trace(go.Scatter(x=match.overs_bowled, y=match.leverage, mode="lines", name="Leverage",
                             line=dict(color=COLORS[summary["model"]], width=2),
                             hovertemplate="%{y:.1f}x"))
    fig.update_yaxes(rangemode="tozero")
    selected = selected_ball(match, "leverage")
    fig.add_vline(x=match.overs_bowled[match.ball_no == selected].iloc[0],
                  line_dash="dash", line_color=MUTED, line_width=1)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    chosen = ball_slider(match, "leverage", selected)
    a, b = st.columns(2)
    a.metric("Leverage", f"{chosen.leverage:.1f}x")
    b.metric("Win probability", f"{chosen['wp_' + summary['model']]:.0%}")

    st.subheader("Most valuable over-breaks in this match")
    breaks = match.groupby("over").first().reset_index().nlargest(5, "leverage")
    st.table(pd.DataFrame({
        "Break before over": breaks.over,
        "Situation": [equation(row) for row in breaks.itertuples()],
        "Win probability": [f"{value:.0%}" for value in breaks[f"wp_{summary['model']}"]],
        "Leverage": [f"{value:.1f}x" for value in breaks.leverage],
    }).set_index("Break before over"))

    by_over = pd.Series(summary["mean_leverage_by_over"])
    peak = match.loc[match.leverage.idxmax()]
    st.markdown(
        "**Observations.**\n"
        f"- This match peaks at {peak.leverage:.1f}x in over {int(peak.over)}.\n"
        f"- Average leverage is {by_over.iloc[6:15].mean():.1f}x before overs 7-15 "
        f"and {by_over.iloc[-1]:.1f}x before the final over.\n"
        f"- {summary['top_10pct_share']:.0%} of all leverage sits in the top 10% of over-breaks.")
    st.info("**Conclusion.** Match tension is concentrated in a small number of over-breaks, mostly late in "
            "close chases. Those breaks are the candidates for premium ad slots.")
    st.markdown("**Limitation.** Leverage is a proxy for viewer attention. "
                "It has not been validated against viewership data.")


def main():
    df = load_predictions()
    metrics = load_json("reports/metrics.json")

    st.title("IPL Chase Win Probability")
    st.caption("Ball-by-ball win probability for IPL run chases from three models, "
               "and the leverage of every moment in the match.")
    overview(df, metrics)
    match = pick_match(df)

    leverage_summary = load_json("models/leverage.json")
    win_probability, leverage = st.tabs(["Win probability", "Leverage"])
    with win_probability:
        win_probability_tab(match, metrics, leverage_summary["model"])
    with leverage:
        leverage_tab(match, leverage_summary)
    st.caption("Data: [Cricsheet](https://cricsheet.org), every IPL match from 2008 to 2026.")


if __name__ == "__main__":
    main()
