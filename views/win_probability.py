"""Win probability page: the three models ball by ball, and how they score on unseen seasons."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import (COLORS, MODELS, MUTED, add_situation_hover, ball_slider, load_json, load_predictions,
                    mark_selected_ball, match_chart, match_summary, selected_ball)

NAIVE = "No model (always back the chasing team)"


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
    mark_selected_ball(fig, match, selected)

    # The top of the chart belongs to the chasing team, the bottom to the defending team.
    for y, anchor, text in ((1, "top", f"▲ {first.batting_team} winning"),
                            (0, "bottom", f"▼ {first.bowling_team} winning")):
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=y, yanchor=anchor, xanchor="left",
                           text=text, showarrow=False, font=dict(color=MUTED))
    fig.update_yaxes(range=[-10, 110], tickvals=[0, 25, 50, 75, 100])
    return fig


match = st.session_state["match"]
metrics = load_json("reports/metrics.json")
main_model = load_json("models/leverage.json")["model"]
overall, by_phase, seasons = metrics["overall"], metrics["by_phase"], metrics["test_seasons"]

st.title("IPL win probability model")
st.write("The chasing team's chance of winning, predicted before every ball by three models.")

df = load_predictions()
matches, deliveries, models, tested = st.columns(4)
matches.metric("Matches", f"{df.match_id.nunique():,}")
deliveries.metric("Deliveries", f"{len(df):,}")
models.metric("Models compared", len(MODELS))
tested.metric("Tested on", f"{seasons[0]}-{seasons[-1]}")

st.subheader("Ball by ball")
st.write(match_summary(match))
selected = selected_ball(match, "wp")
st.plotly_chart(win_probability_chart(match, main_model, selected), width="stretch")
chosen = ball_slider(match, "wp", selected)
for column, (name, label) in zip(st.columns(len(MODELS)), MODELS.items()):
    column.metric(label, f"{chosen[f'wp_{name}']:.0%}")
st.caption("Win probability is predicted from runs needed, balls remaining, wickets in hand, run rates, "
           "the target and how high-scoring the ground is.")

st.subheader("Model comparison")
st.caption(f"Scored on {metrics['test_matches']} matches from {seasons[0]}-{seasons[-1]}, "
           "which no model saw during training.")
labels = {"naive": NAIVE, **MODELS}
st.dataframe(pd.DataFrame({
    "Model": [labels[name] for name in overall],
    "Accuracy": [f"{scores['accuracy']:.1%}" for scores in overall.values()],
    "Log loss": [f"{scores['log_loss']:.3f}" for scores in overall.values()],
    "Brier score": [f"{scores['brier']:.3f}" for scores in overall.values()],
}), width="stretch", hide_index=True)
st.caption("Accuracy: share of balls where the model's favourite went on to win. "
           "Brier score: average squared gap between the predicted probability and the result. "
           "Log loss: like the Brier score, but confident wrong predictions are penalised more heavily.")

st.subheader("Accuracy by phase of the chase")
st.dataframe(pd.DataFrame({
    "Model": list(MODELS.values()),
    "Overs 1-6": [f"{by_phase[name]['powerplay']['accuracy']:.1%}" for name in MODELS],
    "Overs 7-15": [f"{by_phase[name]['middle']['accuracy']:.1%}" for name in MODELS],
    "Overs 16-20": [f"{by_phase[name]['death']['accuracy']:.1%}" for name in MODELS],
}), width="stretch", hide_index=True)

st.subheader("Observations")
accuracies = [overall[name]["accuracy"] for name in MODELS]
st.markdown(
    f"- All three models reach {min(accuracies):.0%}-{max(accuracies):.0%} accuracy, "
    f"against {overall['naive']['accuracy']:.0%} with no model.\n"
    f"- Logistic regression and the neural network are level on log loss "
    f"({overall['logistic']['log_loss']:.3f} and {overall['mlp']['log_loss']:.3f}); "
    f"LightGBM is behind ({overall['lightgbm']['log_loss']:.3f}).\n"
    f"- Accuracy rises through the chase: {by_phase['logistic']['powerplay']['accuracy']:.0%} in overs 1-6 "
    f"to {by_phase['logistic']['death']['accuracy']:.0%} in overs 16-20 for logistic regression.")
st.info("**Conclusion.** The simplest model performs as well as the most complex one. "
        "Logistic regression is used to calculate leverage.", icon=":material/lightbulb:")
