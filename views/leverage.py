"""Leverage page: how much each ball could swing the match, and which over-breaks matter most."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import (COLORS, MUTED, add_situation_hover, ball_slider, equation, load_json, mark_selected_ball,
                    match_chart, match_summary, selected_ball)

match = st.session_state["match"]
summary = load_json("models/leverage.json")
model = summary["model"]

st.title("Leverage")
st.write("How much the next ball can change the win probability, compared with an average ball.")

st.subheader("Ball by ball")
st.write(match_summary(match))
selected = selected_ball(match, "leverage")
fig = match_chart("Leverage (1.0 = average ball)", height=320)
fig.add_hline(y=1, line_dash="dot", line_color=MUTED, line_width=1)
add_situation_hover(fig, match, match.leverage)
fig.add_trace(go.Scatter(x=match.overs_bowled, y=match.leverage, mode="lines", name="Leverage",
                         line=dict(color=COLORS[model], width=2), hovertemplate="%{y:.1f}x"))
mark_selected_ball(fig, match, selected)
fig.update_yaxes(rangemode="tozero")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, width="stretch")

chosen = ball_slider(match, "leverage", selected)
leverage, win_probability = st.columns(2)
leverage.metric("Leverage", f"{chosen.leverage:.1f}x")
win_probability.metric("Win probability", f"{chosen[f'wp_{model}']:.0%}")
st.caption("Leverage is the expected change in win probability from the next ball, "
           "relative to an average ball (1.0 = average).")

st.subheader("Most valuable over-breaks in this match")
breaks = match.groupby("over").first().reset_index().nlargest(5, "leverage")
st.dataframe(pd.DataFrame({
    "Break before over": breaks.over,
    "Situation": [equation(row) for row in breaks.itertuples()],
    "Win probability": [f"{value:.0%}" for value in breaks[f"wp_{model}"]],
    "Leverage": [f"{value:.1f}x" for value in breaks.leverage],
}), width="stretch", hide_index=True)
st.caption("An over-break is valued by the leverage at the start of the over that follows it.")

st.subheader("Observations")
by_over = pd.Series(summary["mean_leverage_by_over"])
peak = match.loc[match.leverage.idxmax()]
st.markdown(
    f"- This match peaks at {peak.leverage:.1f}x in over {int(peak.over)}.\n"
    f"- Average leverage is {by_over.iloc[6:15].mean():.1f}x before overs 7-15 "
    f"and {by_over.iloc[-1]:.1f}x before the final over.\n"
    f"- {summary['top_10pct_share']:.0%} of all leverage sits in the top 10% of over-breaks.")
st.info("**Conclusion.** Match tension is concentrated in a small number of over-breaks, mostly late in "
        "close chases. Those breaks are the candidates for premium ad slots.", icon=":material/lightbulb:")
st.caption("Limitation: leverage is a proxy for viewer attention. "
           "It has not been validated against viewership data.")
