"""
Knicks Odds Analyzer — Full Dashboard
Loads pre-computed RF results from knicks_results.csv (no live model training)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e8f0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.06em;
}
h1 { font-size: 3.2rem !important; color: #ffffff !important; }
h2 { font-size: 1.9rem !important; color: #ff6b2b !important; }
h3 { font-size: 1.3rem !important; color: #ffffff !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #1e1e35;
    border-top: 2px solid #ff6b2b;
    border-radius: 6px;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b6b8a !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.1rem !important;
    color: #ffffff !important;
    letter-spacing: 0.04em;
}

[data-testid="stSelectbox"] > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 6px;
    color: #e8e8f0 !important;
}

[data-testid="stTabs"] button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.1em;
    color: #6b6b8a !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff6b2b !important;
    border-bottom-color: #ff6b2b !important;
}

.card {
    background: linear-gradient(135deg, #13131f 0%, #16162a 100%);
    border: 1px solid #1e1e35;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.game-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
    color: #ffffff;
}
.game-sub { font-size: 0.82rem; color: #6b6b8a; margin-top: 2px; }
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b6b8a;
    margin-bottom: 3px;
}
.stat-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    color: #ffffff;
    letter-spacing: 0.03em;
}
.pill {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 99px;
    margin-right: 6px;
}
.pill-win     { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.pill-loss    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.pill-home    { background: rgba(255,107,43,0.15);  color: #ff6b2b; border: 1px solid rgba(255,107,43,0.3); }
.pill-away    { background: rgba(107,107,138,0.15); color: #9090b0; border: 1px solid rgba(107,107,138,0.3); }
.pill-correct { background: rgba(74,222,128,0.15);  color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.pill-wrong   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #ff6b2b;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e35;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
.divider { border: none; border-top: 1px solid #1e1e35; margin: 16px 0; }
hr { border-color: #1e1e35 !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

BET_AMOUNT     = 20
START_BANKROLL = 1000

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0f", plot_bgcolor="#13131f",
    font=dict(color="#9090b0", family="DM Sans"),
    margin=dict(t=40, b=40, l=40, r=20),
)

# ── LOAD PRE-COMPUTED RESULTS ─────────────────────────────────────────────────

@st.cache_data
def load_results():
    df = pd.read_csv("knicks_results.csv")
    # Ensure correct types
    for col in ["ML Win Prob %", "Poly Win Prob %", "Poly Error",
                "Actual Margin", "Poly Margin", "Poly Spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

try:
    results = load_results()
    data_ok = True
except FileNotFoundError:
    data_ok = False

if not data_ok:
    st.error("**knicks_results.csv not found.**\n\nGenerate it in Colab by running:\n\n```python\nresults.to_csv('knicks_results.csv', index=False)\n```\n\nThen upload it to your GitHub repo alongside app.py.")
    st.stop()

# ── BETTING SIMULATION ────────────────────────────────────────────────────────

def compute_payout(bet, win_prob_pct):
    p = max(min(win_prob_pct / 100.0, 0.99), 0.01)
    return round(bet * (1.0 / p - 1), 2)

@st.cache_data
def run_simulation(_df):
    def sim(df, predict_col):
        bankroll = START_BANKROLL
        history  = [bankroll]
        bets_placed = bets_won = 0
        log = []
        for _, row in df.iterrows():
            predicted_win = row[predict_col] > 50
            actual_win    = row["Result"] == "WIN"
            mkt_prob      = row["Poly Win Prob %"]
            if predicted_win:
                bets_placed += 1
                if actual_win:
                    profit    = compute_payout(BET_AMOUNT, mkt_prob)
                    bets_won += 1
                    outcome   = "WIN"
                else:
                    profit  = -BET_AMOUNT
                    outcome = "LOSS"
                bankroll = round(max(bankroll + profit, 0), 2)
            else:
                profit, outcome = 0.0, "SKIP"
            history.append(bankroll)
            log.append({
                "Date": row["Date"], "Opponent": row["Opponent"],
                "Result": row["Result"], "Pred Prob %": row[predict_col],
                "Mkt Prob %": mkt_prob,
                "Bet?": "YES" if predicted_win else "no",
                "Outcome": outcome,
                "Profit ($)": round(profit, 2) if predicted_win else 0.0,
                "Bankroll ($)": bankroll,
            })
        roi      = round(((bankroll - START_BANKROLL) / START_BANKROLL) * 100, 2)
        win_rate = round(bets_won / bets_placed * 100, 1) if bets_placed else 0
        return {
            "final": bankroll,
            "profit": round(bankroll - START_BANKROLL, 2),
            "roi": roi, "placed": bets_placed, "won": bets_won,
            "lost": bets_placed - bets_won, "win_rate": win_rate,
            "skipped": len(df) - bets_placed,
            "wagered": round(bets_placed * BET_AMOUNT, 2),
            "history": history, "log": pd.DataFrame(log),
        }
    return sim(_df, "ML Win Prob %"), sim(_df, "Poly Win Prob %")

rf_sim, poly_sim = run_simulation(results)

# ── SHARED METRICS ────────────────────────────────────────────────────────────

n            = len(results)
ml_acc       = (results["ML W/L"]   == "Correct").mean() * 100
poly_acc     = (results["Poly W/L"] == "Correct").mean() * 100
poly_avg_err = results["Poly Error"].mean()
ml_correct   = (results["ML W/L"]   == "Correct").sum()
poly_correct = (results["Poly W/L"] == "Correct").sum()

# ── CHART HELPERS ─────────────────────────────────────────────────────────────

def make_gauge(value, title, color="#ff6b2b"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title=dict(text=title, font=dict(size=13, color="#9090b0")),
        number=dict(suffix="%", font=dict(size=28, color="#ffffff",
                                           family="Bebas Neue")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#2a2a40",
                      tickfont=dict(color="#6b6b8a")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#13131f", bordercolor="#1e1e35",
            steps=[dict(range=[0, 50], color="#0a0a0f"),
                   dict(range=[50, 100], color="#13131f")],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.75, value=value),
        )
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      margin=dict(t=30, b=10, l=20, r=20))
    return fig

def color_wl(val):
    if val == "Correct": return "color:#4ade80"
    if val == "Wrong": return "color:#f87171"
    return ""

def color_result(val):
    if val == "WIN":  return "color:#4ade80; font-weight:600"
    if val == "LOSS": return "color:#f87171; font-weight:600"
    return ""

def color_outcome(val):
    if val == "WIN":  return "color:#4ade80; font-weight:600"
    if val == "LOSS": return "color:#f87171; font-weight:600"
    return ""

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family:Bebas Neue,sans-serif; font-size:2.4rem;
                color:#ff6b2b; letter-spacing:0.06em; margin-bottom:2px;'>
                🏀 KNICKS</div>
    <div style='font-family:Bebas Neue,sans-serif; font-size:0.9rem;
                color:#6b6b8a; letter-spacing:0.2em; margin-bottom:28px;'>
                ODDS ANALYZER</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊 Dashboard",
        "🔍 Game Lookup",
        "🤖 RF vs Polymarket",
        "💰 Betting Simulator",
        "📈 Charts",
        "📖 How It Works",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#3a3a5a; line-height:1.8;'>
    <span style='color:#ff6b2b; font-weight:600;'>MODEL</span><br>
    Random Forest (rolling window)<br>
    Trained on 5 prior seasons<br><br>
    <span style='color:#ff6b2b; font-weight:600;'>TEST DATA</span><br>
    2025-26 season · 70 games<br>
    Polymarket odds
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown("""
    <h1>KNICKS ODDS ANALYZER</h1>
    <div style='font-family:DM Sans; font-size:0.95rem; color:#6b6b8a;
                letter-spacing:0.08em; text-transform:uppercase; margin-bottom:32px;'>
        Random Forest ML vs Polymarket — 2025-26 Season
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Games",             n)
    c2.metric("RF W/L Accuracy",   f"{ml_acc:.1f}%")
    c3.metric("Poly W/L Accuracy", f"{poly_acc:.1f}%")
    c4.metric("Poly Avg Err",      f"{poly_avg_err:.1f} pts")
    c5.metric("RF Betting Profit",
              f"${rf_sim['profit']:+,.0f}", f"ROI {rf_sim['roi']:+.1f}%")
    c6.metric("Poly Betting Profit",
              f"${poly_sim['profit']:+,.0f}", f"ROI {poly_sim['roi']:+.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown("## ACCURACY")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(make_gauge(ml_acc, "Random Forest", "#ff6b2b"),
                            use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge(poly_acc, "Polymarket", "#4fa3e0"),
                            use_container_width=True)

    with col_right:
        st.markdown("## W/L BREAKDOWN")
        ml_c = (results["ML W/L"]   == "Correct").sum()
        ml_w = (results["ML W/L"]   == "Wrong").sum()
        po_c = (results["Poly W/L"] == "Correct").sum()
        po_w = (results["Poly W/L"] == "Wrong").sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Correct", x=["Random Forest", "Polymarket"], y=[ml_c, po_c],
            marker_color=["#ff6b2b", "#4fa3e0"],
            text=[ml_c, po_c], textposition="inside",
            textfont=dict(color="#fff", size=14),
        ))
        fig.add_trace(go.Bar(
            name="Wrong", x=["Random Forest", "Polymarket"], y=[ml_w, po_w],
            marker_color=["#2a1a0f", "#0f1a2a"],
            text=[ml_w, po_w], textposition="inside",
            textfont=dict(color="#888", size=14),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=220,
                          showlegend=True, margin=dict(t=10, b=30, l=10, r=10),
                          legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                      font=dict(color="#e8e8f0")))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## RESULTS TABLE")

    display_cols = ["Date", "Opponent", "Home/Away", "Result", "Actual Margin",
                    "ML Win Prob %", "ML W/L",
                    "Poly Win Prob %", "Poly Margin", "Poly Error", "Poly W/L"]
    styled = (results[display_cols].style
              .applymap(color_result, subset=["Result"])
              .applymap(color_wl, subset=["ML W/L", "Poly W/L"])
              .format({"Actual Margin": "{:+.0f}", "Poly Margin": "{:+.1f}",
                       "ML Win Prob %": "{:.1f}%", "Poly Win Prob %": "{:.1f}%",
                       "Poly Error": "{:.1f}"})
              .set_properties(**{"background-color": "#13131f",
                                 "color": "#e8e8f0", "border-color": "#1e1e35"}))
    st.dataframe(styled, use_container_width=True, height=420)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GAME LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Game Lookup":
    st.markdown("""
    <h1>GAME LOOKUP</h1>
    <div style='color:#6b6b8a; font-size:0.9rem; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:28px;'>
        Select any game to compare RF vs Polymarket
    </div>
    """, unsafe_allow_html=True)

    opponents  = sorted(results["Opponent"].unique())
    col_a, col_b = st.columns([1, 2])
    with col_a:
        sel_opp = st.selectbox("Opponent", opponents)
    opp_games = results[results["Opponent"] == sel_opp].reset_index(drop=True)
    with col_b:
        if len(opp_games) > 1:
            labels  = [f"{r['Date']} — {r['Home/Away']} ({r['Result']})"
                       for _, r in opp_games.iterrows()]
            sel_idx = st.selectbox("Game", range(len(labels)),
                                    format_func=lambda i: labels[i])
        else:
            sel_idx = 0
    row = opp_games.iloc[sel_idx]

    st.markdown("---")

    win    = row["Result"] == "WIN"
    rc     = "#4ade80" if win else "#f87171"
    ha_cls = "pill-home" if row["Home/Away"] == "Home" else "pill-away"
    wl_cls = "pill-win"  if win else "pill-loss"

    # Safely get Training Games if column exists
    train_games_html = ""
    if "Training Games" in row.index:
        train_games_html = f"""
            <div>
                <div class="stat-label">Training Games</div>
                <div class="stat-val">{int(row['Training Games'])}</div>
            </div>"""

    st.markdown(f"""
    <div class="card" style="border-left:3px solid #ff6b2b;">
        <div class="game-title">Knicks vs {row['Opponent']}</div>
        <div class="game-sub" style="margin-bottom:16px;">
            {row['Date']} &nbsp;·&nbsp;
            <span class="pill {ha_cls}">{row['Home/Away']}</span>
            <span class="pill {wl_cls}">{row['Result']}</span>
        </div>
        <div style="display:flex; gap:40px; flex-wrap:wrap;">
            <div>
                <div class="stat-label">Final Score</div>
                <div class="stat-val">NYK {row['Knicks Score']} — {row['Opp Score']}</div>
            </div>
            <div>
                <div class="stat-label">Actual Margin</div>
                <div class="stat-val" style="color:{rc};">{row['Actual Margin']:+.0f}</div>
            </div>
            <div>
                <div class="stat-label">Total Points</div>
                <div class="stat-val">{row['Actual Total']}</div>
            </div>
            {train_games_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    ml_ok  = row["ML W/L"]   == "Correct"
    po_ok  = row["Poly W/L"] == "Correct"
    ml_cls = "pill-correct" if ml_ok else "pill-wrong"
    po_cls = "pill-correct" if po_ok else "pill-wrong"
    ml_txt = "✓ Correct"    if ml_ok else "✗ Wrong"
    po_txt = "✓ Correct"    if po_ok else "✗ Wrong"

    with c1:
        st.markdown(f"""
        <div class="card" style="border-top:2px solid #ff6b2b;">
            <div class="section-label">Random Forest ML</div>
            <div class="stat-label">Win Probability</div>
            <div class="stat-val" style="font-size:2.8rem;">{row['ML Win Prob %']}%</div>
            <div style="margin-top:10px;">
                <div class="stat-label">Prediction</div>
                <div style="font-size:1rem; color:#e8e8f0; margin-top:2px;">
                    {'Knicks WIN' if row['ML Win Prob %'] > 50 else 'Knicks LOSS'}
                </div>
            </div>
            <div style="margin-top:14px;">
                <span class="pill {ml_cls}">{ml_txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card" style="border-top:2px solid #4fa3e0;">
            <div class="section-label">Polymarket</div>
            <div class="stat-label">Win Probability</div>
            <div class="stat-val" style="font-size:2.8rem; color:#4fa3e0;">{row['Poly Win Prob %']}%</div>
            <div style="display:flex; gap:28px; margin-top:10px; flex-wrap:wrap;">
                <div>
                    <div class="stat-label">Spread</div>
                    <div style="font-size:1rem; color:#e8e8f0;">{row['Poly Spread']:+.1f}</div>
                </div>
                <div>
                    <div class="stat-label">Implied Margin</div>
                    <div style="font-size:1rem; color:#e8e8f0;">{row['Poly Margin']:+.1f}</div>
                </div>
                <div>
                    <div class="stat-label">Margin Error</div>
                    <div style="font-size:1rem; color:#e8e8f0;">{row['Poly Error']} pts</div>
                </div>
            </div>
            <div style="margin-top:14px;">
                <span class="pill {po_cls}">{po_txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Random Forest", "Polymarket", "Actual Outcome"],
        y=[row["ML Win Prob %"], row["Poly Win Prob %"], 100 if win else 0],
        marker_color=["#ff6b2b", "#4fa3e0", "#4ade80" if win else "#f87171"],
        text=[f"{row['ML Win Prob %']:.1f}%", f"{row['Poly Win Prob %']:.1f}%",
              "WIN" if win else "LOSS"],
        textposition="outside", textfont=dict(color="#e8e8f0", size=13),
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#6b6b8a", opacity=0.5,
                  annotation_text="50% threshold",
                  annotation_font_color="#6b6b8a")
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text="Win Probability Comparison",
                                 font=dict(color="#e8e8f0", size=14)),
                      yaxis=dict(range=[0, 120], gridcolor="#1e1e35",
                                 title="Win Probability (%)"),
                      xaxis=dict(gridcolor="#1e1e35"))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RF VS POLYMARKET
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 RF vs Polymarket":
    st.markdown("""
    <h1>RF vs POLYMARKET</h1>
    <div style='color:#6b6b8a; font-size:0.9rem; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:28px;'>
        Head-to-head accuracy comparison
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["ACCURACY", "DISAGREEMENTS", "FULL TABLE"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RF Correct",   f"{ml_correct}/{n}",   f"{ml_acc:.1f}%")
        c2.metric("Poly Correct", f"{poly_correct}/{n}", f"{poly_acc:.1f}%")
        diff = ml_acc - poly_acc
        c3.metric("RF Edge", f"{diff:+.1f}%",
                  "RF leads" if diff > 0 else "Poly leads")
        both_right = ((results["ML W/L"] == "Correct") &
                      (results["Poly W/L"] == "Correct")).sum()
        c4.metric("Both Correct", f"{both_right}/{n}")

        st.markdown("---")

        # Cumulative accuracy over time
        res = results.copy()
        res["ML_cum"]   = (res["ML W/L"]   == "Correct").cumsum()
        res["Poly_cum"] = (res["Poly W/L"] == "Correct").cumsum()
        res["game_num"] = range(1, len(res) + 1)
        res["ML_pct"]   = res["ML_cum"]   / res["game_num"] * 100
        res["Poly_pct"] = res["Poly_cum"] / res["game_num"] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res["Date"], y=res["ML_pct"], name="Random Forest",
            line=dict(color="#ff6b2b", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=res["Date"], y=res["Poly_pct"], name="Polymarket",
            line=dict(color="#4fa3e0", width=2.5),
        ))
        fig.add_hline(y=50, line_dash="dot", line_color="#6b6b8a", opacity=0.4)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          title=dict(text="Cumulative W/L Accuracy Over Season",
                                     font=dict(color="#e8e8f0", size=14)),
                          yaxis=dict(gridcolor="#1e1e35", title="Accuracy (%)",
                                     range=[40, 85]),
                          xaxis=dict(gridcolor="#1e1e35", tickangle=-35),
                          legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                      font=dict(color="#e8e8f0")))
        st.plotly_chart(fig, use_container_width=True)

        # Per-opponent accuracy
        opp_stats = results.groupby("Opponent").agg(
            RF_acc=("ML W/L",   lambda x: (x == "Correct").mean() * 100),
            Poly_acc=("Poly W/L", lambda x: (x == "Correct").mean() * 100),
        ).reset_index().sort_values("RF_acc", ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="RF",
                               x=opp_stats["Opponent"], y=opp_stats["RF_acc"],
                               marker_color="#ff6b2b"))
        fig2.add_trace(go.Bar(name="Polymarket",
                               x=opp_stats["Opponent"], y=opp_stats["Poly_acc"],
                               marker_color="#4fa3e0"))
        fig2.update_layout(**PLOTLY_LAYOUT, barmode="group", height=380,
                            title=dict(text="Accuracy by Opponent",
                                       font=dict(color="#e8e8f0", size=14)),
                            yaxis=dict(gridcolor="#1e1e35", title="Accuracy (%)"),
                            xaxis=dict(gridcolor="#1e1e35", tickangle=-40),
                            legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                        font=dict(color="#e8e8f0")))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        rf_right = results[(results["ML W/L"] == "Correct") &
                            (results["Poly W/L"] == "Wrong")]
        po_right = results[(results["Poly W/L"] == "Correct") &
                            (results["ML W/L"] == "Wrong")]
        both_w   = results[(results["ML W/L"] == "Wrong") &
                            (results["Poly W/L"] == "Wrong")]

        cols = ["Date", "Opponent", "Result", "Actual Margin",
                "ML Win Prob %", "Poly Win Prob %"]

        st.markdown(f"### RF RIGHT, POLY WRONG — {len(rf_right)} games")
        if len(rf_right):
            st.dataframe(rf_right[cols].style
                         .applymap(color_result, subset=["Result"])
                         .set_properties(**{"background-color": "#13131f",
                                            "color": "#e8e8f0",
                                            "border-color": "#1e1e35"}),
                         use_container_width=True)

        st.markdown(f"### POLY RIGHT, RF WRONG — {len(po_right)} games")
        if len(po_right):
            st.dataframe(po_right[cols].style
                         .applymap(color_result, subset=["Result"])
                         .set_properties(**{"background-color": "#13131f",
                                            "color": "#e8e8f0",
                                            "border-color": "#1e1e35"}),
                         use_container_width=True)

        st.markdown(f"### BOTH WRONG — {len(both_w)} games")
        if len(both_w):
            st.dataframe(both_w[cols].style
                         .applymap(color_result, subset=["Result"])
                         .set_properties(**{"background-color": "#13131f",
                                            "color": "#e8e8f0",
                                            "border-color": "#1e1e35"}),
                         use_container_width=True)

    with tab3:
        cols = ["Date", "Opponent", "Home/Away", "Result", "Actual Margin",
                "ML Win Prob %", "ML W/L",
                "Poly Win Prob %", "Poly Margin", "Poly Error", "Poly W/L"]
        st.dataframe(
            results[cols].style
            .applymap(color_wl,     subset=["ML W/L", "Poly W/L"])
            .applymap(color_result, subset=["Result"])
            .format({"Actual Margin": "{:+.0f}", "Poly Margin": "{:+.1f}",
                     "ML Win Prob %": "{:.1f}%", "Poly Win Prob %": "{:.1f}%",
                     "Poly Error": "{:.1f}"})
            .set_properties(**{"background-color": "#13131f",
                               "color": "#e8e8f0", "border-color": "#1e1e35"}),
            use_container_width=True, height=500,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BETTING SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Betting Simulator":
    st.markdown("""
    <h1>BETTING SIMULATOR</h1>
    <div style='color:#6b6b8a; font-size:0.9rem; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:28px;'>
        $20 flat bet on every predicted Knicks win · Payout at Polymarket odds
    </div>
    """, unsafe_allow_html=True)

    rf_pc = "#4ade80" if rf_sim["profit"]   >= 0 else "#f87171"
    po_pc = "#4ade80" if poly_sim["profit"] >= 0 else "#f87171"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card" style="border-top:2px solid #ff6b2b;">
            <div class="section-label">Random Forest Strategy</div>
            <div style="display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <div class="stat-label">Final Bankroll</div>
                    <div class="stat-val">${rf_sim['final']:,.0f}</div>
                </div>
                <div>
                    <div class="stat-label">Net Profit</div>
                    <div class="stat-val" style="color:{rf_pc};">
                        ${rf_sim['profit']:+,.0f}
                    </div>
                </div>
                <div>
                    <div class="stat-label">ROI</div>
                    <div class="stat-val" style="color:{rf_pc};">
                        {rf_sim['roi']:+.1f}%
                    </div>
                </div>
            </div>
            <hr class="divider">
            <div style="display:flex; gap:20px; flex-wrap:wrap;
                        font-size:0.85rem; color:#9090b0;">
                <span>Bets: <b style="color:#e8e8f0;">{rf_sim['placed']}</b></span>
                <span>Won: <b style="color:#4ade80;">{rf_sim['won']}</b></span>
                <span>Lost: <b style="color:#f87171;">{rf_sim['lost']}</b></span>
                <span>Win rate: <b style="color:#e8e8f0;">{rf_sim['win_rate']}%</b></span>
                <span>Skipped: <b style="color:#e8e8f0;">{rf_sim['skipped']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card" style="border-top:2px solid #4fa3e0;">
            <div class="section-label">Polymarket Strategy</div>
            <div style="display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <div class="stat-label">Final Bankroll</div>
                    <div class="stat-val">${poly_sim['final']:,.0f}</div>
                </div>
                <div>
                    <div class="stat-label">Net Profit</div>
                    <div class="stat-val" style="color:{po_pc};">
                        ${poly_sim['profit']:+,.0f}
                    </div>
                </div>
                <div>
                    <div class="stat-label">ROI</div>
                    <div class="stat-val" style="color:{po_pc};">
                        {poly_sim['roi']:+.1f}%
                    </div>
                </div>
            </div>
            <hr class="divider">
            <div style="display:flex; gap:20px; flex-wrap:wrap;
                        font-size:0.85rem; color:#9090b0;">
                <span>Bets: <b style="color:#e8e8f0;">{poly_sim['placed']}</b></span>
                <span>Won: <b style="color:#4ade80;">{poly_sim['won']}</b></span>
                <span>Lost: <b style="color:#f87171;">{poly_sim['lost']}</b></span>
                <span>Win rate: <b style="color:#e8e8f0;">{poly_sim['win_rate']}%</b></span>
                <span>Skipped: <b style="color:#e8e8f0;">{poly_sim['skipped']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## BANKROLL OVER TIME")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(rf_sim["history"]))), y=rf_sim["history"],
        name=f"Random Forest ({rf_sim['roi']:+.1f}%)",
        line=dict(color="#ff6b2b", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,43,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(poly_sim["history"]))), y=poly_sim["history"],
        name=f"Polymarket ({poly_sim['roi']:+.1f}%)",
        line=dict(color="#4fa3e0", width=2.5),
        fill="tozeroy", fillcolor="rgba(79,163,224,0.05)",
    ))
    fig.add_hline(y=START_BANKROLL, line_dash="dot", line_color="#6b6b8a",
                  opacity=0.5, annotation_text=f"Starting ${START_BANKROLL}",
                  annotation_font_color="#6b6b8a")
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      yaxis=dict(gridcolor="#1e1e35", title="Bankroll ($)",
                                 tickprefix="$", tickformat=",.0f"),
                      xaxis=dict(gridcolor="#1e1e35", title="Game Number"),
                      legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                  font=dict(color="#e8e8f0")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## PER-BET PROFIT / LOSS")
    col_rf, col_po = st.columns(2)

    def pnl_chart(strat, color, title):
        bets = strat["log"][strat["log"]["Bet?"] == "YES"].copy()
        bets["Cumulative"] = bets["Profit ($)"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(bets))), y=bets["Profit ($)"],
            marker_color=["#4ade80" if p > 0 else "#f87171"
                          for p in bets["Profit ($)"]],
            opacity=0.8, name="Per bet",
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(bets))), y=bets["Cumulative"],
            line=dict(color=color, width=2), name="Cumulative",
        ))
        fig.add_hline(y=0, line_color="#6b6b8a", line_width=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          title=dict(text=title, font=dict(color="#e8e8f0", size=13)),
                          yaxis=dict(gridcolor="#1e1e35", title="Profit ($)",
                                     tickprefix="$"),
                          xaxis=dict(gridcolor="#1e1e35", title="Bet #"),
                          legend=dict(bgcolor="#13131f",
                                      font=dict(color="#e8e8f0")))
        return fig

    with col_rf:
        st.plotly_chart(pnl_chart(rf_sim,   "#ff6b2b", "Random Forest — P&L per bet"),
                        use_container_width=True)
    with col_po:
        st.plotly_chart(pnl_chart(poly_sim, "#4fa3e0", "Polymarket — P&L per bet"),
                        use_container_width=True)

    st.markdown("---")
    st.markdown("## GAME LOGS")
    tab_rf, tab_po, tab_h2h = st.tabs(["RF BETS", "POLYMARKET BETS", "HEAD TO HEAD"])

    log_cols = ["Date", "Opponent", "Result", "Pred Prob %",
                "Mkt Prob %", "Outcome", "Profit ($)", "Bankroll ($)"]

    with tab_rf:
        rf_log = rf_sim["log"][rf_sim["log"]["Bet?"] == "YES"]
        st.dataframe(
            rf_log[log_cols].style
            .applymap(color_outcome, subset=["Outcome", "Result"])
            .format({"Pred Prob %": "{:.1f}%", "Mkt Prob %": "{:.1f}%",
                     "Profit ($)": "${:+.2f}", "Bankroll ($)": "${:.2f}"})
            .set_properties(**{"background-color": "#13131f",
                               "color": "#e8e8f0", "border-color": "#1e1e35"}),
            use_container_width=True, height=420,
        )

    with tab_po:
        po_log = poly_sim["log"][poly_sim["log"]["Bet?"] == "YES"]
        st.dataframe(
            po_log[log_cols].style
            .applymap(color_outcome, subset=["Outcome", "Result"])
            .format({"Pred Prob %": "{:.1f}%", "Mkt Prob %": "{:.1f}%",
                     "Profit ($)": "${:+.2f}", "Bankroll ($)": "${:.2f}"})
            .set_properties(**{"background-color": "#13131f",
                               "color": "#e8e8f0", "border-color": "#1e1e35"}),
            use_container_width=True, height=420,
        )

    with tab_h2h:
        h2h = pd.DataFrame({
            "Date":           rf_sim["log"]["Date"].values,
            "Opponent":       rf_sim["log"]["Opponent"].values,
            "Actual":         rf_sim["log"]["Result"].values,
            "RF Bet?":        rf_sim["log"]["Bet?"].values,
            "RF Outcome":     rf_sim["log"]["Outcome"].values,
            "RF Profit ($)":  rf_sim["log"]["Profit ($)"].values,
            "Poly Bet?":      poly_sim["log"]["Bet?"].values,
            "Poly Outcome":   poly_sim["log"]["Outcome"].values,
            "Poly Profit ($)":poly_sim["log"]["Profit ($)"].values,
        })
        disagreed = (h2h["RF Bet?"] != h2h["Poly Bet?"]).sum()
        st.caption(f"Games where RF and Polymarket gave different predictions: {disagreed}")
        st.dataframe(
            h2h.style.set_properties(**{"background-color": "#13131f",
                                         "color": "#e8e8f0",
                                         "border-color": "#1e1e35"}),
            use_container_width=True, height=420,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHARTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Charts":
    st.markdown("""
    <h1>SEASON CHARTS</h1>
    <div style='color:#6b6b8a; font-size:0.9rem; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:28px;'>Visual season breakdown</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "WIN PROBABILITIES", "MARGIN ANALYSIS",
        "POLYMARKET ERROR",  "DISTRIBUTIONS",
    ])

    with tab1:
        win_mask = results["Result"] == "WIN"
        fig = go.Figure()
        for mask, label, color in [(win_mask, "WIN", "#4ade80"),
                                    (~win_mask, "LOSS", "#f87171")]:
            fig.add_trace(go.Scatter(
                x=results.loc[mask, "ML Win Prob %"],
                y=results.loc[mask, "Poly Win Prob %"],
                mode="markers", name=label,
                marker=dict(color=color, size=9, opacity=0.8,
                            line=dict(color="#2a2a40", width=1)),
                text=results.loc[mask, "Opponent"],
            ))
        fig.add_vline(x=50, line_dash="dot", line_color="#6b6b8a", opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="#6b6b8a", opacity=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=dict(text="RF Win Prob vs Polymarket Win Prob — colored by result",
                                     font=dict(color="#e8e8f0", size=13)),
                          xaxis=dict(gridcolor="#1e1e35", title="RF Win Probability (%)"),
                          yaxis=dict(gridcolor="#1e1e35", title="Polymarket Win Probability (%)"),
                          legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                      font=dict(color="#e8e8f0")))
        st.plotly_chart(fig, use_container_width=True)

        # RF calibration
        bins = pd.cut(results["ML Win Prob %"], bins=[0, 40, 50, 60, 70, 80, 100])
        cal  = results.groupby(bins, observed=True).agg(
            actual_win_rate=("Result", lambda x: (x == "WIN").mean() * 100),
            count=("Result", "count"),
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=cal["ML Win Prob %"].astype(str), y=cal["actual_win_rate"],
            marker_color="#ff6b2b",
            text=[f"{v:.0f}% ({c}g)" for v, c in
                  zip(cal["actual_win_rate"], cal["count"])],
            textposition="outside", textfont=dict(color="#e8e8f0"),
        ))
        fig2.add_hline(y=50, line_dash="dot", line_color="#6b6b8a", opacity=0.5)
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                            title=dict(text="RF Calibration: predicted prob bucket vs actual win rate",
                                       font=dict(color="#e8e8f0", size=13)),
                            yaxis=dict(gridcolor="#1e1e35", title="Actual Win Rate (%)"),
                            xaxis=dict(gridcolor="#1e1e35", title="RF Predicted Prob Bucket"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        wins   = results[results["Result"] == "WIN"]["Actual Margin"]
        losses = results[results["Result"] == "LOSS"]["Actual Margin"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=wins,   name="WIN",
                                    marker_color="#4ade80", opacity=0.75, nbinsx=15))
        fig.add_trace(go.Histogram(x=losses, name="LOSS",
                                    marker_color="#f87171", opacity=0.75, nbinsx=15))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=360,
                           title=dict(text="Distribution of Actual Margins",
                                      font=dict(color="#e8e8f0", size=13)),
                           xaxis=dict(gridcolor="#1e1e35", title="Margin (pts)"),
                           yaxis=dict(gridcolor="#1e1e35", title="Games"),
                           legend=dict(bgcolor="#13131f", font=dict(color="#e8e8f0")))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=results["Poly Margin"], y=results["Actual Margin"],
            mode="markers",
            marker=dict(
                color=["#4ade80" if r == "WIN" else "#f87171"
                       for r in results["Result"]],
                size=8, opacity=0.7, line=dict(color="#2a2a40", width=1),
            ),
            text=results["Opponent"],
        ))
        m = max(abs(results["Poly Margin"].max()), abs(results["Actual Margin"].max())) + 5
        fig2.add_trace(go.Scatter(x=[-m, m], y=[-m, m], mode="lines",
                                   line=dict(color="#6b6b8a", dash="dot", width=1),
                                   name="Perfect prediction"))
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                            title=dict(text="Polymarket Implied Margin vs Actual Margin",
                                       font=dict(color="#e8e8f0", size=13)),
                            xaxis=dict(gridcolor="#1e1e35", title="Poly Implied Margin"),
                            yaxis=dict(gridcolor="#1e1e35", title="Actual Margin"),
                            legend=dict(bgcolor="#13131f", font=dict(color="#e8e8f0")))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["Date"], y=results["Poly Error"],
            mode="lines+markers", line=dict(color="#4fa3e0", width=2),
            marker=dict(size=5), fill="tozeroy",
            fillcolor="rgba(79,163,224,0.07)",
        ))
        fig.add_hline(y=results["Poly Error"].mean(), line_dash="dot",
                      line_color="#ff6b2b", opacity=0.7,
                      annotation_text=f"Avg: {results['Poly Error'].mean():.1f} pts",
                      annotation_font_color="#ff6b2b")
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                           title=dict(text="Polymarket Margin Error Per Game",
                                      font=dict(color="#e8e8f0", size=13)),
                           xaxis=dict(gridcolor="#1e1e35", tickangle=-35),
                           yaxis=dict(gridcolor="#1e1e35", title="Error (pts)"))
        st.plotly_chart(fig, use_container_width=True)

        opp_err = (results.groupby("Opponent")["Poly Error"]
                   .mean().sort_values(ascending=False))
        avg_err = opp_err.mean()
        fig2 = go.Figure(go.Bar(
            x=opp_err.index, y=opp_err.values,
            marker_color=["#f87171" if e > avg_err else "#4fa3e0"
                          for e in opp_err.values],
        ))
        fig2.add_hline(y=avg_err, line_dash="dot", line_color="#ff6b2b",
                       opacity=0.7, annotation_text="Season avg",
                       annotation_font_color="#ff6b2b")
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                            title=dict(text="Polymarket Avg Margin Error by Opponent",
                                       font=dict(color="#e8e8f0", size=13)),
                            xaxis=dict(gridcolor="#1e1e35", tickangle=-40),
                            yaxis=dict(gridcolor="#1e1e35", title="Avg Error (pts)"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=results["ML Win Prob %"], nbinsx=20,
                                        marker_color="#ff6b2b", opacity=0.8))
            fig.add_vline(x=50, line_dash="dot", line_color="#6b6b8a")
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                               title=dict(text="RF Win Prob Distribution",
                                          font=dict(color="#e8e8f0", size=13)),
                               xaxis=dict(gridcolor="#1e1e35", title="Win Prob (%)"),
                               yaxis=dict(gridcolor="#1e1e35", title="Games"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=results["Poly Win Prob %"], nbinsx=20,
                                         marker_color="#4fa3e0", opacity=0.8))
            fig2.add_vline(x=50, line_dash="dot", line_color="#6b6b8a")
            fig2.update_layout(**PLOTLY_LAYOUT, height=300,
                                title=dict(text="Polymarket Win Prob Distribution",
                                           font=dict(color="#e8e8f0", size=13)),
                                xaxis=dict(gridcolor="#1e1e35", title="Win Prob (%)"),
                                yaxis=dict(gridcolor="#1e1e35", title="Games"))
            st.plotly_chart(fig2, use_container_width=True)

        ha = results.groupby("Home/Away").agg(
            RF_acc=("ML W/L",   lambda x: (x == "Correct").mean() * 100),
            Poly_acc=("Poly W/L", lambda x: (x == "Correct").mean() * 100),
            Win_rate=("Result",  lambda x: (x == "WIN").mean() * 100),
        ).reset_index()
        fig3 = go.Figure()
        for col, color, name in [
            ("RF_acc",   "#ff6b2b", "RF Accuracy"),
            ("Poly_acc", "#4fa3e0", "Poly Accuracy"),
            ("Win_rate", "#4ade80", "Actual Win Rate"),
        ]:
            fig3.add_trace(go.Bar(name=name, x=ha["Home/Away"],
                                   y=ha[col], marker_color=color))
        fig3.update_layout(**PLOTLY_LAYOUT, barmode="group", height=320,
                            title=dict(text="Home vs Away — Accuracy & Win Rate",
                                       font=dict(color="#e8e8f0", size=13)),
                            yaxis=dict(gridcolor="#1e1e35", title="%"),
                            xaxis=dict(gridcolor="#1e1e35"),
                            legend=dict(bgcolor="#13131f", bordercolor="#1e1e35",
                                        font=dict(color="#e8e8f0")))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📖 How It Works":
    st.markdown("<h1>HOW IT WORKS</h1>", unsafe_allow_html=True)

    st.markdown("""
    ## THE MODEL

    The **Random Forest** classifier is trained on 5 full seasons of Knicks game
    data (2020-21 through 2024-25) using a **rolling window** approach: each
    2025-26 prediction is made using only data available at the time of that game.

    For each game, the model is retrained on all 5 prior seasons plus every
    2025-26 game that happened *before* it. By mid-season the model has seen
    how the current Knicks are actually performing, not just historical averages.

    ---

    ## THE 4 FEATURES

    | Feature | Description |
    |---------|-------------|
    | `is_home` | 1 if Knicks at MSG, 0 if away |
    | `knicks_avg_pts` | Knicks historical avg score vs this opponent |
    | `opp_avg_pts` | Opponent historical avg score vs Knicks |
    | `algo_margin` | Predicted margin (avg pts diff ± 2.5 home adj) |

    ---

    ## COMPARING TO POLYMARKET

    Polymarket is a real-money prediction market. It produces a **spread** and
    **win probability** per game. Both the RF and Polymarket output a win
    probability — whoever picks the correct winner is marked correct.

    ---

    ## THE BETTING SIMULATION

    Bets **$20** on every game where the model predicts a Knicks win (>50%).
    Payouts use Polymarket's implied decimal odds:

    > 60% win prob → odds of 1.667 → $20 bet wins $13.33 profit if correct,
    > loses $20 if wrong.

    Higher-odds wins pay more. The simulation rewards finding games where the
    model is more confident than Polymarket.

    ---

    ## LIMITATIONS

    - Only 4 features — no injury data, rest days, or recent form streaks
    - ~600 training games is small for a Random Forest
    - Home advantage fixed at 2.5 pts for all games
    - Polymarket reflects real-money incentives and live information the model lacks
    """)
