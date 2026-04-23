import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Knicks Odds Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111;
    border-right: 2px solid #f97316;
}

[data-testid="stSidebar"] * {
    color: #f0f0f0 !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

h1 { font-size: 3rem !important; font-weight: 800 !important; }
h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #f97316 !important; }
h3 { font-size: 1.3rem !important; font-weight: 600 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-top: 3px solid #f97316;
    border-radius: 4px;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f0f0f0 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 4px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2a2a2a;
}

/* Divider */
hr {
    border-color: #2a2a2a !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f97316 !important;
    border-bottom-color: #f97316 !important;
}

/* Game card */
.game-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 4px solid #f97316;
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.game-card-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #f0f0f0;
    margin-bottom: 4px;
}

.game-card-sub {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 16px;
}

.pred-block {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 14px 18px;
}

.pred-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #666;
    margin-bottom: 8px;
}

.pred-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0f0;
}

.pred-sub {
    font-size: 0.8rem;
    color: #888;
    margin-top: 4px;
}

.winner-tag {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 2px;
    margin-top: 8px;
}

.tag-closer { background: #f97316; color: #000; }
.tag-further { background: #2a2a2a; color: #666; }
.tag-correct { color: #4ade80; }
.tag-wrong { color: #f87171; }

.win-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #f97316;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a2a;
}
</style>
""", unsafe_allow_html=True)

# ── ALGORITHM ─────────────────────────────────────────────────────────────────

HOME_ADVANTAGE = 2.5

OPPONENT_NAME_MAP = {
    "Cleveland Cavaliers":    "Cavaliers",
    "Boston Celtics":         "Celtics",
    "Miami Heat":             "Heat",
    "Milwaukee Bucks":        "Bucks",
    "Chicago Bulls":          "Bulls",
    "Washington Wizards":     "Wizards",
    "Minnesota Timberwolves": "Timberwolves",
    "Brooklyn Nets":          "Nets",
    "Memphis Grizzlies":      "Grizzlies",
    "Orlando Magic":          "Magic",
    "Dallas Mavericks":       "Mavericks",
    "Charlotte Hornets":      "Hornets",
    "Toronto Raptors":        "Raptors",
    "Utah Jazz":              "Jazz",
    "Philadelphia 76ers":     "76ers",
    "Indiana Pacers":         "Pacers",
    "Atlanta Hawks":          "Hawks",
    "New Orleans Pelicans":   "Pelicans",
    "San Antonio Spurs":      "Spurs",
    "Detroit Pistons":        "Pistons",
    "Los Angeles Clippers":   "Clippers",
    "Phoenix Suns":           "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings":       "Kings",
    "Golden State Warriors":  "Warriors",
    "Los Angeles Lakers":     "Lakers",
    "Denver Nuggets":         "Nuggets",
    "Houston Rockets":        "Rockets",
    "Oklahoma City Thunder":  "Thunder",
}

@st.cache_data
def load_season(filepath):
    df = pd.read_excel(filepath)
    df["is_home"]       = df["Unnamed: 5"].isna()
    df["opp_short"]     = df["Opponent"].map(OPPONENT_NAME_MAP)
    df["total"]         = df["Tm"] + df["Opp"]
    df["knicks_margin"] = df["Tm"] - df["Opp"]
    df["date_parsed"]   = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_polymarket(filepath):
    df = pd.read_excel(filepath)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

@st.cache_data
def build_algorithm(training_df):
    knicks_scored   = training_df.groupby("opp_short")["Tm"].mean()
    knicks_allowed  = training_df.groupby("opp_short")["Opp"].mean()
    default_scored  = training_df["Tm"].mean()
    default_allowed = training_df["Opp"].mean()
    return knicks_scored, knicks_allowed, default_scored, default_allowed

def predict(opponent, is_home, knicks_scored, knicks_allowed, default_scored, default_allowed):
    k   = knicks_scored.get(opponent, default_scored)
    o   = knicks_allowed.get(opponent, default_allowed)
    adj = HOME_ADVANTAGE if is_home else -HOME_ADVANTAGE
    return round(k + o + adj, 1), round((k - o) + adj, 1)

@st.cache_data
def build_results(_poly_df, _actual_df, _knicks_scored, _knicks_allowed, default_scored, default_allowed):
    rows = []
    for _, p in _poly_df.iterrows():
        opponent  = p["opponent"]
        game_date = p["game_date"]
        candidates = _actual_df[
            (_actual_df["opp_short"] == opponent) &
            (abs(_actual_df["date_parsed"] - game_date) <= pd.Timedelta(days=30))
        ].copy()
        if candidates.empty:
            continue
        candidates["date_diff"] = abs(candidates["date_parsed"] - game_date)
        game = candidates.sort_values("date_diff").iloc[0]

        actual_total  = int(game["total"])
        actual_margin = float(game["knicks_margin"])
        is_home       = bool(game["is_home"])
        actual_win    = actual_margin > 0

        algo_total, algo_margin = predict(
            opponent, is_home, _knicks_scored, _knicks_allowed, default_scored, default_allowed
        )

        poly_spread   = float(p["implied_spread"])
        poly_margin   = round(-poly_spread, 1)
        poly_win_prob = round(p["knicks_win_prob"] * 100, 1)

        algo_error = round(abs(actual_margin - algo_margin), 1)
        poly_error = round(abs(actual_margin - poly_margin), 1)

        if algo_error < poly_error:
            closer = "Algorithm"
        elif poly_error < algo_error:
            closer = "Polymarket"
        else:
            closer = "Tie"

        rows.append({
            "Date":            game["date_parsed"].strftime("%b %d, %Y"),
            "date_sort":       game["date_parsed"],
            "Opponent":        opponent,
            "Home/Away":       "Home" if is_home else "Away",
            "Knicks Score":    int(game["Tm"]),
            "Opp Score":       int(game["Opp"]),
            "Actual Total":    actual_total,
            "Actual Margin":   actual_margin,
            "Algo Total":      algo_total,
            "Algo Margin":     algo_margin,
            "Algo Error":      algo_error,
            "Algo W/L":        "Correct" if (algo_margin > 0) == actual_win else "Wrong",
            "Poly Spread":     poly_spread,
            "Poly Win Prob":   poly_win_prob,
            "Poly Margin":     poly_margin,
            "Poly Error":      poly_error,
            "Poly W/L":        "Correct" if (poly_margin > 0) == actual_win else "Wrong",
            "Closer":          closer,
            "Result":          "WIN" if actual_win else "LOSS",
        })
    return pd.DataFrame(rows).sort_values("date_sort").reset_index(drop=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

try:
    df_train = pd.concat([
        load_season("Knicks2023.xlsx"),
        load_season("Knicks2024.xlsx"),
    ], ignore_index=True)
    df_actual = load_season("Knicks2025.xlsx")
    poly      = load_polymarket("knicks_polymarket_odds.xlsx")

    knicks_scored, knicks_allowed, default_scored, default_allowed = build_algorithm(df_train)
    results = build_results(poly, df_actual, knicks_scored, knicks_allowed, default_scored, default_allowed)
    data_loaded = True

except FileNotFoundError as e:
    data_loaded = False
    missing_file = str(e)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family: Barlow Condensed, sans-serif; font-size: 2rem;
                font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;
                color: #f97316; margin-bottom: 4px;'>🏀 Knicks</div>
    <div style='font-family: Barlow Condensed, sans-serif; font-size: 1rem;
                text-transform: uppercase; letter-spacing: 0.12em; color: #666;
                margin-bottom: 24px;'>Odds Analyzer</div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigate**")
    page = st.radio(
        "",
        ["📊 Dashboard", "🔍 Game Lookup", "📈 Charts", "📖 How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #444; line-height: 1.6;'>
    <b style='color:#666;'>Training data</b><br>
    2023-24 season<br>
    2024-25 season<br><br>
    <b style='color:#666;'>Test data</b><br>
    2025-26 season<br>
    Polymarket odds
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if not data_loaded:
    st.error(f"Could not load data files. Make sure all four Excel files are in the same folder as app.py.\n\n{missing_file}")
    st.stop()

# ── PAGE: DASHBOARD ───────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    st.markdown("""
    <h1 style='color:#f0f0f0; margin-bottom:0;'>Knicks Odds</h1>
    <div style='font-family: Barlow Condensed, sans-serif; font-size:1.1rem;
                text-transform:uppercase; letter-spacing:0.15em; color:#f97316;
                margin-bottom:32px;'>Algorithm vs Polymarket — 2025-26 Season</div>
    """, unsafe_allow_html=True)

    n             = len(results)
    algo_avg_err  = results["Algo Error"].mean()
    poly_avg_err  = results["Poly Error"].mean()
    algo_win_pct  = (results["Algo W/L"] == "Correct").mean() * 100
    poly_win_pct  = (results["Poly W/L"] == "Correct").mean() * 100
    algo_closer   = (results["Closer"] == "Algorithm").sum()
    poly_closer   = (results["Closer"] == "Polymarket").sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Games", n)
    col2.metric("Algo Avg Error", f"{algo_avg_err:.1f} pts")
    col3.metric("Poly Avg Error", f"{poly_avg_err:.1f} pts")
    col4.metric("Algo W/L Acc.", f"{algo_win_pct:.0f}%")
    col5.metric("Poly W/L Acc.", f"{poly_win_pct:.0f}%")
    col6.metric("Poly Closer", f"{poly_closer} games", delta=f"Algo: {algo_closer}", delta_color="off")

    st.markdown("---")

    # Recent games table
    st.markdown("## Recent Games")
    display_cols = ["Date", "Opponent", "Home/Away", "Result",
                    "Actual Margin", "Algo Margin", "Algo Error",
                    "Poly Margin", "Poly Error", "Closer"]

    def color_closer(val):
        if val == "Algorithm":
            return "color: #f97316; font-weight: 600;"
        elif val == "Polymarket":
            return "color: #60a5fa; font-weight: 600;"
        return ""

    def color_result(val):
        if val == "WIN":
            return "color: #4ade80; font-weight: 600;"
        return "color: #f87171; font-weight: 600;"

    styled = (
        results[display_cols]
        .style
        .applymap(color_closer, subset=["Closer"])
        .applymap(color_result, subset=["Result"])
        .format({"Actual Margin": "{:+.0f}", "Algo Margin": "{:+.1f}",
                 "Poly Margin": "{:+.1f}", "Algo Error": "{:.1f}", "Poly Error": "{:.1f}"})
        .set_properties(**{"background-color": "#1a1a1a", "color": "#f0f0f0",
                           "border-color": "#2a2a2a"})
    )
    st.dataframe(styled, use_container_width=True, height=400)

# ── PAGE: GAME LOOKUP ─────────────────────────────────────────────────────────

elif page == "🔍 Game Lookup":
    st.markdown("""
    <h1 style='color:#f0f0f0; margin-bottom:4px;'>Game Lookup</h1>
    <div style='font-family: Barlow Condensed, sans-serif; color:#666;
                text-transform:uppercase; letter-spacing:0.12em;
                font-size:0.9rem; margin-bottom:32px;'>
        Select any game to compare predictions
    </div>
    """, unsafe_allow_html=True)

    opponents = sorted(results["Opponent"].unique())
    selected_opp = st.selectbox("Select Opponent", opponents)

    opp_games = results[results["Opponent"] == selected_opp].reset_index(drop=True)

    if len(opp_games) > 1:
        game_labels = [f"{row['Date']} — {row['Home/Away']} ({row['Result']})"
                       for _, row in opp_games.iterrows()]
        selected_game_idx = st.selectbox("Select Game", range(len(game_labels)),
                                          format_func=lambda i: game_labels[i])
        row = opp_games.iloc[selected_game_idx]
    else:
        row = opp_games.iloc[0]

    st.markdown("---")

    # Game header
    result_color = "#4ade80" if row["Result"] == "WIN" else "#f87171"
    st.markdown(f"""
    <div class='game-card'>
        <div class='game-card-header'>
            Knicks vs {row['Opponent']}
            <span style='color:{result_color}; margin-left:16px;'>{row['Result']}</span>
        </div>
        <div class='game-card-sub'>{row['Date']} &nbsp;·&nbsp; {row['Home/Away']} &nbsp;·&nbsp;
            NYK {row['Knicks Score']} — {row['Opp Score']} OPP
        </div>
        <div style='display:flex; gap:32px; flex-wrap:wrap;'>
            <div>
                <div style='font-size:0.75rem; color:#666; text-transform:uppercase;
                            letter-spacing:0.1em;'>Actual Margin</div>
                <div style='font-family: Barlow Condensed, sans-serif; font-size:2.5rem;
                            font-weight:800; color:{result_color};'>
                    {row['Actual Margin']:+.0f}
                </div>
            </div>
            <div>
                <div style='font-size:0.75rem; color:#666; text-transform:uppercase;
                            letter-spacing:0.1em;'>Total Points</div>
                <div style='font-family: Barlow Condensed, sans-serif; font-size:2.5rem;
                            font-weight:800; color:#f0f0f0;'>{row['Actual Total']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side predictions
    c1, c2 = st.columns(2)

    algo_closer = row["Closer"] == "Algorithm"
    poly_closer = row["Closer"] == "Polymarket"

    with c1:
        tag = "<span class='winner-tag tag-closer'>✓ Closer</span>" if algo_closer else "<span class='winner-tag tag-further'>Further</span>"
        wl_color = "#4ade80" if row["Algo W/L"] == "Correct" else "#f87171"
        st.markdown(f"""
        <div class='pred-block' style='border-top: 3px solid #f97316;'>
            <div class='pred-label'>Algorithm Prediction</div>
            <div class='pred-value'>{row['Algo Margin']:+.1f}</div>
            <div class='pred-sub'>Predicted margin</div>
            <div class='pred-sub' style='margin-top:4px;'>Total: {row['Algo Total']}</div>
            <div class='pred-sub' style='margin-top:4px;'>Error: <b style='color:#f0f0f0;'>{row['Algo Error']} pts off</b></div>
            <div class='pred-sub' style='margin-top:4px; color:{wl_color};'>W/L: {row['Algo W/L']}</div>
            {tag}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        tag = "<span class='winner-tag tag-closer'>✓ Closer</span>" if poly_closer else "<span class='winner-tag tag-further'>Further</span>"
        wl_color = "#4ade80" if row["Poly W/L"] == "Correct" else "#f87171"
        st.markdown(f"""
        <div class='pred-block' style='border-top: 3px solid #60a5fa;'>
            <div class='pred-label'>Polymarket Prediction</div>
            <div class='pred-value'>{row['Poly Margin']:+.1f}</div>
            <div class='pred-sub'>Implied margin (from spread {row['Poly Spread']:+.1f})</div>
            <div class='pred-sub' style='margin-top:4px;'>Win probability: {row['Poly Win Prob']}%</div>
            <div class='pred-sub' style='margin-top:4px;'>Error: <b style='color:#f0f0f0;'>{row['Poly Error']} pts off</b></div>
            <div class='pred-sub' style='margin-top:4px; color:{wl_color};'>W/L: {row['Poly W/L']}</div>
            {tag}
        </div>
        """, unsafe_allow_html=True)

    # Error bar chart
    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Algorithm", "Polymarket"],
        y=[row["Algo Error"], row["Poly Error"]],
        marker_color=["#f97316", "#60a5fa"],
        text=[f"{row['Algo Error']} pts", f"{row['Poly Error']} pts"],
        textposition="outside",
        textfont=dict(color="#f0f0f0", size=14),
    ))
    fig.update_layout(
        title=dict(text="Margin Error Comparison", font=dict(color="#f0f0f0", size=16)),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(color="#888"),
        yaxis=dict(gridcolor="#1a1a1a", title="Points off from actual margin"),
        xaxis=dict(gridcolor="#1a1a1a"),
        height=300, margin=dict(t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE: CHARTS ──────────────────────────────────────────────────────────────

elif page == "📈 Charts":
    st.markdown("""
    <h1 style='color:#f0f0f0; margin-bottom:4px;'>Season Charts</h1>
    <div style='font-family: Barlow Condensed, sans-serif; color:#666;
                text-transform:uppercase; letter-spacing:0.12em;
                font-size:0.9rem; margin-bottom:32px;'>Visual breakdown of algorithm vs Polymarket</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Error Over Time", "Per-Opponent Breakdown", "Error Distribution"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["Date"], y=results["Algo Error"],
            mode="lines+markers", name="Algorithm",
            line=dict(color="#f97316", width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=results["Date"], y=results["Poly Error"],
            mode="lines+markers", name="Polymarket",
            line=dict(color="#60a5fa", width=2),
            marker=dict(size=5),
        ))
        fig.update_layout(
            title="Margin error per game over the season",
            paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
            font=dict(color="#888"),
            yaxis=dict(gridcolor="#1a1a1a", title="Pts off"),
            xaxis=dict(gridcolor="#1a1a1a", tickangle=-45),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a"),
            height=400, margin=dict(t=40, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        opp_summary = results.groupby("Opponent").agg(
            Algo_Error=("Algo Error", "mean"),
            Poly_Error=("Poly Error", "mean"),
            Games=("Opponent", "count"),
        ).reset_index().sort_values("Algo_Error")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Algorithm", x=opp_summary["Opponent"], y=opp_summary["Algo_Error"],
            marker_color="#f97316",
        ))
        fig.add_trace(go.Bar(
            name="Polymarket", x=opp_summary["Opponent"], y=opp_summary["Poly_Error"],
            marker_color="#60a5fa",
        ))
        fig.update_layout(
            barmode="group",
            title="Average margin error by opponent",
            paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
            font=dict(color="#888"),
            yaxis=dict(gridcolor="#1a1a1a", title="Avg pts off"),
            xaxis=dict(gridcolor="#1a1a1a", tickangle=-45),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a"),
            height=450, margin=dict(t=40, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results["Algo Error"], name="Algorithm",
            marker_color="#f97316", opacity=0.7, nbinsx=20,
        ))
        fig.add_trace(go.Histogram(
            x=results["Poly Error"], name="Polymarket",
            marker_color="#60a5fa", opacity=0.7, nbinsx=20,
        ))
        fig.update_layout(
            barmode="overlay",
            title="Distribution of margin errors",
            paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
            font=dict(color="#888"),
            xaxis=dict(gridcolor="#1a1a1a", title="Error (pts)"),
            yaxis=dict(gridcolor="#1a1a1a", title="Number of games"),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a"),
            height=400, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Closer breakdown pie
        c1, c2 = st.columns(2)
        closer_counts = results["Closer"].value_counts()
        with c1:
            fig2 = go.Figure(go.Pie(
                labels=closer_counts.index,
                values=closer_counts.values,
                marker=dict(colors=["#f97316", "#60a5fa", "#444"]),
                hole=0.5,
                textfont=dict(color="#f0f0f0"),
            ))
            fig2.update_layout(
                title="Who was closer per game?",
                paper_bgcolor="#0d0d0d",
                font=dict(color="#888"),
                legend=dict(bgcolor="#1a1a1a"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        wl_algo = results["Algo W/L"].value_counts()
        wl_poly = results["Poly W/L"].value_counts()
        with c2:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name="Algorithm", x=["Correct", "Wrong"],
                y=[wl_algo.get("Correct", 0), wl_algo.get("Wrong", 0)],
                marker_color=["#4ade80", "#f87171"],
            ))
            fig3.add_trace(go.Bar(
                name="Polymarket", x=["Correct", "Wrong"],
                y=[wl_poly.get("Correct", 0), wl_poly.get("Wrong", 0)],
                marker_color=["#4ade80", "#f87171"],
                opacity=0.5,
            ))
            fig3.update_layout(
                barmode="group",
                title="W/L prediction accuracy",
                paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                font=dict(color="#888"),
                yaxis=dict(gridcolor="#1a1a1a"),
                legend=dict(bgcolor="#1a1a1a"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: HOW IT WORKS ────────────────────────────────────────────────────────

elif page == "📖 How It Works":
    st.markdown("""
    <h1 style='color:#f0f0f0; margin-bottom:4px;'>How It Works</h1>
    <div style='font-family: Barlow Condensed, sans-serif; color:#666;
                text-transform:uppercase; letter-spacing:0.12em;
                font-size:0.9rem; margin-bottom:32px;'>Algorithm explained</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### The Algorithm

    The algorithm is trained on two full seasons of Knicks game data (2023-24 and 2024-25).
    For each opponent, it calculates:

    - **Average points the Knicks scored** against that opponent across both seasons
    - **Average points that opponent scored** against the Knicks across both seasons

    These two averages are then combined with a **home/away adjustment of ±2.5 points**
    to produce a predicted total score and a predicted margin.

    ---

    ### The Formula

    ```
    Predicted Total  = Avg Knicks pts vs opponent
                     + Avg opponent pts vs Knicks
                     ± 2.5 (home advantage)

    Predicted Margin = Avg Knicks pts vs opponent
                     − Avg opponent pts vs Knicks
                     ± 2.5 (home advantage)
    ```

    A **positive margin** means the algorithm expects the Knicks to win.
    A **negative margin** means it expects them to lose.

    ---

    ### Comparing to Polymarket

    Polymarket is a prediction market where people bet real money on game outcomes.
    It produces a **spread** for each game — for example, -7.5 means the Knicks are
    favored to win by 7.5 points.

    To compare the two on equal footing, the spread is converted to an implied margin:
    a spread of -7.5 becomes a predicted Knicks margin of **+7.5**.

    ---

    ### Measuring Accuracy

    For each game, the **margin error** is calculated for both predictions:

    ```
    Error = |Actual margin − Predicted margin|
    ```

    Whichever prediction had the smaller error is marked as "closer."
    Win/loss accuracy is also tracked — did each method correctly pick the winner?

    ---

    ### Limitations

    - Treats all historical games equally (a game from 2 years ago = last week)
    - Does not account for injuries, trades, or roster changes
    - Uses only Knicks-specific data — no knowledge of how opponents played others
    - Fixed 2.5 pt home advantage for every game
    """)

    st.markdown("---")
    st.markdown("### Training Data Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training games", len(df_train))
    c2.metric("Avg Knicks score (training)", f"{df_train['Tm'].mean():.1f}")
    c3.metric("Avg opp score (training)", f"{df_train['Opp'].mean():.1f}")

    st.markdown("### Per-Opponent Training Averages")
    opp_table = pd.DataFrame({
        "Opponent": knicks_scored.index,
        "Avg Knicks Pts": knicks_scored.values.round(1),
        "Avg Opp Pts": knicks_allowed.reindex(knicks_scored.index).values.round(1),
    }).sort_values("Opponent").reset_index(drop=True)
    opp_table["Avg Margin"] = (opp_table["Avg Knicks Pts"] - opp_table["Avg Opp Pts"]).round(1)
    st.dataframe(opp_table, use_container_width=True, height=400)
