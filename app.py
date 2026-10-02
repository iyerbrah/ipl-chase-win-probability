"""Entry point. Run with: streamlit run app.py"""
import pandas as pd
import streamlit as st

from charts import load_predictions

# Short team names keep the match picker readable in the sidebar.
SHORT_NAMES = {
    "Chennai Super Kings": "CSK", "Deccan Chargers": "DEC", "Delhi Capitals": "DC", "Delhi Daredevils": "DD",
    "Gujarat Lions": "GL", "Gujarat Titans": "GT", "Kings XI Punjab": "KXIP", "Kochi Tuskers Kerala": "KTK",
    "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG", "Mumbai Indians": "MI",
    "Pune Warriors": "PWI", "Punjab Kings": "PBKS", "Rajasthan Royals": "RR",
    "Rising Pune Supergiant": "RPS", "Rising Pune Supergiants": "RPS",
    "Royal Challengers Bangalore": "RCB", "Royal Challengers Bengaluru": "RCB", "Sunrisers Hyderabad": "SRH",
}

st.set_page_config(page_title="IPL Win Probability", page_icon=":material/sports_cricket:")


def pick_match(df):
    """Season and match pickers in the sidebar, shared by both pages."""
    season = st.selectbox("Season", sorted(df.season.unique(), reverse=True))
    matches = (df[df.season == season].groupby("match_id")
               .agg(date=("date", "first"), batting=("batting_team", "first"),
                    bowling=("bowling_team", "first"), peak=("leverage", "max"))
               .sort_values("date", ascending=False))
    labels = {mid: f"{pd.Timestamp(m.date):%d %b}  ·  {SHORT_NAMES[m.bowling]} v {SHORT_NAMES[m.batting]}"
              for mid, m in matches.iterrows()}
    tightest = list(labels).index(matches.peak.idxmax())   # open on the season's closest finish
    match_id = st.selectbox("Match", list(labels), index=tightest, format_func=labels.get)

    match = df[df.match_id == match_id].sort_values("ball_no").copy()
    match["overs_bowled"] = (120 - match.balls_remaining) / 6
    return match


pages = [
    st.Page("views/win_probability.py", title="Win probability", icon=":material/show_chart:", default=True),
    st.Page("views/leverage.py", title="Leverage", icon=":material/bolt:", url_path="Leverage"),
]
page = st.navigation(pages)

df = load_predictions()
with st.sidebar:
    st.subheader("Match")
    st.session_state["match"] = pick_match(df)

    st.subheader("About")
    st.write("Three models predict the chasing team's chance of winning before every ball of an IPL chase. "
             "They were trained on 2008 to 2024 and tested on 2025 and 2026.")
    st.write("The **Leverage** page shows which moments of a match matter most.")
    st.caption(f"Data: Cricsheet, {df.match_id.nunique():,} matches, {len(df):,} deliveries.")

page.run()
