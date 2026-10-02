"""Data loading and the chart pieces both pages use."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODELS = {"logistic": "Logistic regression", "lightgbm": "LightGBM", "mlp": "Neural network"}
# Fixed colour per model, the same in every chart.
COLORS = {"logistic": "#3987e5", "lightgbm": "#d95926", "mlp": "#199e70"}
MUTED = "#898781"


@st.cache_data
def load_predictions():
    return pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")


@st.cache_data
def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text())


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


def match_summary(match):
    first = match.iloc[0]
    return (f"**{first.batting_team}** needed **{int(first.target)}** to beat {first.bowling_team}, "
            f"and **{'won' if first.chase_won else 'lost'}**.")


def match_chart(y_title, height):
    """Empty chart with overs along the bottom and one hover box for all lines."""
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        xaxis=dict(title="Overs bowled in the chase", range=[0, 20], dtick=2),
        yaxis=dict(title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def add_situation_hover(fig, match, y):
    """Invisible trace that puts the match situation at the top of the hover box."""
    fig.add_trace(go.Scatter(x=match.overs_bowled, y=y, mode="lines", line=dict(width=0), showlegend=False,
                             customdata=[situation(row) for row in match.itertuples()],
                             hovertemplate="%{customdata}<extra></extra>"))
    fig.update_xaxes(unifiedhovertitle=dict(text="Match situation"))


def slider_key(match, page):
    return f"{page}_ball_{match.match_id.iloc[0]}"


def selected_ball(match, page):
    """The ball chosen on this page's slider (drawn below the chart); starts at the tensest ball."""
    default = int(match.ball_no[match.leverage.idxmax()])
    return st.session_state.get(slider_key(match, page), default)


def mark_selected_ball(fig, match, selected):
    fig.add_vline(x=match.overs_bowled[match.ball_no == selected].iloc[0],
                  line_dash="dash", line_color=MUTED, line_width=1)


def ball_slider(match, page, selected):
    """Slider to pick a ball; shows the score at that moment and returns that ball's row."""
    selected = st.select_slider("Pick a moment in the chase (overs bowled)", options=list(match.ball_no),
                                value=selected, format_func=lambda ball: over_notation(
                                    match.balls_remaining[match.ball_no == ball].iloc[0]),
                                key=slider_key(match, page))
    chosen = match[match.ball_no == selected].iloc[0]
    score = int(chosen.target - chosen.runs_needed)
    st.write(f"**{chosen.batting_team} {score}/{10 - int(chosen.wickets_in_hand)}.** {situation(chosen)}.")
    return chosen
