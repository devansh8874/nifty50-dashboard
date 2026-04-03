# ==============================================================================
#  NIFTY 50 ALGO TRADING DASHBOARD — Streamlit Cloud App  [app.py]
#  MBA Dissertation: AI-Driven Algorithmic Trading System — SVNIT Surat
# ------------------------------------------------------------------------------
#  ARCHITECTURE (why this file does ZERO computation):
#
#    GitHub Actions (daily 6:30 AM IST)          Streamlit Cloud
#    ─────────────────────────────────           ──────────────────────────────
#    01_data_updater.py  →  OHLCV CSVs           app.py  ← THIS FILE
#    02_ml_model.py      →  signals +            Just reads pre-built CSVs.
#                           forecasts +          Loads in < 3 seconds.
#                           indicators           Zero RAM spike. No timeouts.
#    git commit 02_data/ → auto-redeploy ──────►
#
#  DEPLOY STEPS:
#    1. Run pipeline once locally → 02_data/*.csv files are generated
#    2. git add 02_data/*.csv && git commit && git push
#    3. share.streamlit.io → New app → set Main file = app.py
#    4. Add .github/workflows/update_data.yml (provided) → auto-updates daily
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ── All 50 NIFTY symbols — shown even before pipeline runs ───────────────────
ALL_50_SYMBOLS = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL",
    "CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL",
    "GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
    "HINDUNILVR","ICICIBANK","ITC","INFY","INDIGO",
    "JSWSTEEL","JIOFIN","KOTAKBANK","LT","M&M",
    "MARUTI","MAXHEALTH","NTPC","NESTLEIND","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SHRIRAMFIN","SBIN",
    "SUNPHARMA","TCS","TATACONSUM","TATAMOTORS","TATASTEEL",
    "TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO",
]

DATA_DIR     = "02_data"
SIGNALS_CSV  = f"{DATA_DIR}/latest_signals_summary.csv"
ML_CSV       = f"{DATA_DIR}/ml_forecasts.csv"
IND_CSV      = f"{DATA_DIR}/nifty50_with_indicators.csv"
BT_CSV       = f"{DATA_DIR}/backtest_results.csv"
LAST_RUN_TXT = f"{DATA_DIR}/.last_run"

# ==============================================================================
# PAGE CONFIG + CSS
# ==============================================================================

st.set_page_config(
    page_title="NIFTY 50 | Algo Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
.data-banner {
    background:#1a2744; border-left:4px solid #2196f3;
    padding:10px 16px; border-radius:4px; margin-bottom:10px;
    font-size:0.85rem; color:#90caf9;
}
</style>""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING — reads only, cached 1 hour
# ==============================================================================

@st.cache_data(ttl=3600)
def load_all():
    out = {}
    for key, path, parse_dates in [
        ("signals",    SIGNALS_CSV, False),
        ("ml",         ML_CSV,      False),
        ("indicators", IND_CSV,     ["date"]),
        ("backtest",   BT_CSV,      False),
    ]:
        out[key] = pd.read_csv(path, parse_dates=parse_dates,
                               low_memory=False) if os.path.exists(path) else pd.DataFrame()
    out["last_run"] = open(LAST_RUN_TXT).read().strip() if os.path.exists(LAST_RUN_TXT) else None
    return out

data       = load_all()
signals_df = data["signals"]
ml_df      = data["ml"]
ind_df     = data["indicators"]
bt_df      = data["backtest"]
last_run   = data["last_run"]


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.title("📊 NIFTY 50 Dashboard")

# Data freshness
if not signals_df.empty:
    as_of = signals_df["as_of_date"].iloc[0] if "as_of_date" in signals_df.columns else "—"
    st.sidebar.success(f"✅ Signals as of: **{as_of}**")
    if last_run:
        st.sidebar.caption(f"Pipeline ran: {last_run}")
else:
    st.sidebar.error("⚠️ No data. See setup instructions below.")

st.sidebar.markdown("---")

# Stock selector — ranked by signal score, full 50 always shown
if not signals_df.empty and "symbol" in signals_df.columns:
    ranked    = signals_df.sort_values("score", ascending=False)["symbol"].tolist()
    remaining = [s for s in ALL_50_SYMBOLS if s not in ranked]
    all_syms  = ranked + remaining
    st.sidebar.caption(f"Ranked by signal score · {len(ranked)} stocks loaded")
else:
    all_syms = ALL_50_SYMBOLS
    st.sidebar.caption("All 50 NIFTY stocks · awaiting data")

symbol = st.sidebar.selectbox("🔍 Select Stock", all_syms)
st.sidebar.markdown("---")

# Chart controls
st.sidebar.markdown("### ⚙️ Chart Controls")
candles = st.sidebar.slider("Historical candles", 60, 500, 180, step=20)
forecast_days = st.sidebar.select_slider(
    "ML Forecast horizon",
    options=[5, 21, 42, 63, 84],
    value=84,
    format_func=lambda x: {
        5:"1 week (D+5)", 21:"1 month (D+21)", 42:"2 months (D+42)",
        63:"3 months (D+63)", 84:"4 months (D+84)",
    }[x],
)
st.sidebar.markdown("---")
show_forecast = st.sidebar.checkbox("📈 ML forecast lines", value=True)
show_bb       = st.sidebar.checkbox("📊 Bollinger Bands",   value=True)
show_sma200   = st.sidebar.checkbox("📉 SMA 200",           value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 Trading Rules")
st.sidebar.markdown("""
- **Capital** ₹1,00,000 · **Per trade** 20%  
- **Max open** 2 · **RR** 1:2  
- Equity delivery · **No shorting**
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Indicators")
st.sidebar.markdown("""
1. RSI (14) — 60/40 zones  
2. SMA 20/50 — crossover  
3. SMA 200 — trend baseline  
4. OBV — volume confirm  
5. Bollinger Bands — volatility
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Ensemble")
st.sidebar.markdown("""
- **ARIMA** — univariate close  
- **Prophet** — close + RSI + Vol  
- **XGBoost** — close + 12 indicators  
- **Consensus** — median of 3
""")


# ==============================================================================
# HEADER
# ==============================================================================

c_title, c_btn = st.columns([5, 1])
with c_title:
    st.title(f"📈 {symbol} — Technical Analysis + ML Forecast")
with c_btn:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Data freshness notice
if not ind_df.empty and "date" in ind_df.columns:
    max_date = pd.to_datetime(ind_df["date"]).max()
    days_old = (pd.Timestamp.now() - max_date).days
    if days_old <= 2:
        st.markdown(
            f'<div class="data-banner">📅 Data current as of '
            f'<b>{max_date.date()}</b> · Updated daily via GitHub Actions</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            f"⚠️ Data is {days_old} days old (last: {max_date.date()}). "
            "Check your GitHub Actions tab — the daily workflow may need attention."
        )
else:
    st.error(
        "📂 **No data found in `02_data/`.** "
        "Run the pipeline locally first, commit the output CSVs, then push to GitHub."
    )

st.markdown("---")


# ==============================================================================
# SIGNAL METRICS ROW
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Signal",   str(sig.get("signal",          "N/A")))
    c2.metric("Score",    str(sig.get("score",           "N/A")))
    c3.metric("Price",    f"₹{float(sig.get('close',0)):,.2f}")
    c4.metric("RSI",      f"{float(sig.get('rsi',50)):.1f}")
    c5.metric("OBV",      "↑ Rising" if sig.get("obv_rising") else "↓ Falling")
    c6.metric("LT Trend", str(sig.get("long_term_trend", "N/A")))


# ==============================================================================
# TRADE PLAN (BUY / STRONG BUY only)
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]
    if str(sig.get("signal", "")) in ("BUY", "STRONG BUY"):
        st.markdown("---")
        st.subheader("💰 Trade Plan")
        tp1, tp2, tp3, tp4, tp5 = st.columns(5)
        tp1.metric("Entry",     f"₹{sig.get('entry', 0):,}")
        tp2.metric("Stop Loss", f"₹{sig.get('stop_loss', 0):,}  ({sig.get('sl_pct','')})")
        tp3.metric("Target",    f"₹{sig.get('target', 0):,}  ({sig.get('upside_pct','')})")
        tp4.metric("Qty",       str(sig.get("qty", "N/A")))
        tp5.metric("Position",  str(sig.get("position_value", "N/A")))
        st.info(f"📅 **Hold**: {sig.get('hold_duration', 'N/A')}")
        if sig.get("ml_swing_verdict"):
            st.success(f"🤖 **ML Verdict**: {sig['ml_swing_verdict']}")


# ==============================================================================
# MAIN CHART  — Candlestick + Indicators + ML Forecast
# ==============================================================================

st.markdown("---")
label_map = {5:"1 wk",21:"1 mo",42:"2 mo",63:"3 mo",84:"4 mo"}
st.subheader(
    f"📊 {symbol} — Daily Chart + Indicators + "
    f"{label_map.get(forecast_days,'')} ML Forecast (D+{forecast_days})"
)
st.caption(
    "SMA 20 🟠 · SMA 50 🔵 · SMA 200 🔴 · BB ⬜ · "
    "ARIMA 🟣 · Prophet 🩵 · XGBoost 🟠 · Consensus ⬜ thick"
)

if not ind_df.empty and symbol in ind_df["symbol"].values:
    s_df = ind_df[ind_df["symbol"] == symbol].copy()
    s_df["date"] = pd.to_datetime(s_df["date"])
    s_df = s_df.sort_values("date").tail(candles).copy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.60, 0.20, 0.20],
        subplot_titles=[f"{symbol} — Price + ML Forecast", "RSI (14)", "OBV"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=s_df["date"], open=s_df["open"], high=s_df["high"],
        low=s_df["low"],  close=s_df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # SMAs
    for col_name, color, width in [
        ("MA_20","#ff9800",1.5), ("MA_50","#2196f3",1.5)
    ]:
        if col_name in s_df.columns:
            fig.add_trace(go.Scatter(
                x=s_df["date"], y=s_df[col_name],
                name=col_name.replace("_"," "),
                line=dict(color=color, width=width),
            ), row=1, col=1)

    if show_sma200 and "MA_200" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["MA_200"],
            name="SMA 200", line=dict(color="#ff4444", width=1.5, dash="dot"),
        ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_Upper" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.05)", showlegend=False,
        ), row=1, col=1)

    # ML Forecast
    if show_forecast and not ml_df.empty and symbol in ml_df["symbol"].values:
        ml_row       = ml_df[ml_df["symbol"] == symbol].iloc[0]
        last_date    = s_df["date"].max()
        last_close   = float(s_df["close"].iloc[-1])
        future_dates = pd.bdate_range(last_date, periods=forecast_days + 1)[1:]

        model_cfg = [
            ("arima",   "#9c27b0", "dot",     "ARIMA"),
            ("prophet", "#00bcd4", "dash",    "Prophet"),
            ("xgb",     "#ff9800", "dashdot", "XGBoost"),
        ]
        all_vals = {}
        for prefix, color, dash, mlabel in model_cfg:
            vals = [ml_row.get(f"{prefix}_d{d}", np.nan)
                    for d in range(1, forecast_days + 1)]
            valid = [(future_dates[i], v) for i, v in enumerate(vals)
                     if i < len(future_dates) and not pd.isna(v)]
            if not valid:
                continue
            x_l = [last_date] + [p[0] for p in valid]
            y_l = [last_close] + [p[1] for p in valid]
            fig.add_trace(go.Scatter(
                x=x_l, y=y_l, name=mlabel,
                line=dict(color=color, width=1.5, dash=dash),
                mode="lines", opacity=0.80,
            ), row=1, col=1)
            all_vals[prefix] = vals

        # Consensus + uncertainty band
        if len(all_vals) >= 2:
            consensus = []
            for i in range(forecast_days):
                day_v = [v[i] for v in all_vals.values()
                         if i < len(v) and not pd.isna(v[i])]
                consensus.append(float(np.median(day_v)) if day_v else np.nan)

            valid_c = [(future_dates[i], v) for i, v in enumerate(consensus)
                       if i < len(future_dates) and not pd.isna(v)]
            if valid_c:
                xc = [last_date] + [p[0] for p in valid_c]
                yc = [last_close] + [p[1] for p in valid_c]

                # Shaded ±5% band
                x_band = list(xc[1:]) + list(reversed(xc[1:]))
                y_band = [v * 1.05 for v in yc[1:]] + \
                         [v * 0.95 for v in reversed(yc[1:])]
                fig.add_trace(go.Scatter(
                    x=x_band, y=y_band, fill="toself",
                    fillcolor="rgba(255,255,255,0.05)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=xc, y=yc,
                    name=f"Consensus ({forecast_days}d)",
                    line=dict(color="#ffffff", width=2.5),
                    mode="lines",
                ), row=1, col=1)

                # Endpoint annotation
                end_chg = (yc[-1] - last_close) / last_close * 100
                fig.add_annotation(
                    x=xc[-1], y=yc[-1],
                    text=f"  ₹{yc[-1]:,.0f} ({end_chg:+.1f}%)",
                    showarrow=False, font=dict(color="#ffffff", size=11),
                    xanchor="left",
                )

    # RSI
    if "RSI" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["RSI"],
            name="RSI", line=dict(color="#e91e63", width=1.5),
        ), row=2, col=1)
        for lvl, clr in [(60, "rgba(255,100,100,0.4)"), (40, "rgba(100,255,100,0.4)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr, row=2, col=1)
            fig.add_annotation(
                x=s_df["date"].min(), y=lvl, text=str(lvl),
                showarrow=False, xref="x2", yref="y2",
                font=dict(color=clr, size=10),
            )

    # OBV
    if "OBV" in s_df.columns and "OBV_Rising" in s_df.columns:
        obv_c = ["#26a69a" if r else "#ef5350"
                 for r in s_df["OBV_Rising"].fillna(False)]
        fig.add_trace(go.Bar(
            x=s_df["date"], y=s_df["OBV"], name="OBV", marker_color=obv_c,
        ), row=3, col=1)

    fig.update_layout(
        height=780, template="plotly_dark",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_yaxes(gridcolor="#1e1e2e")
    fig.update_xaxes(gridcolor="#1e1e2e")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(
        f"No chart data for **{symbol}**. "
        "Commit `02_data/nifty50_with_indicators.csv` to your GitHub repo."
    )


# ==============================================================================
# SIGNAL BREAKDOWN  +  FORECAST TABLE
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]
    st.markdown("---")
    col_sig, col_ml = st.columns(2)

    with col_sig:
        st.subheader("📋 Signal Breakdown")
        for d in str(sig.get("signal_details", "")).split(" | "):
            d = d.strip()
            if not d:
                continue
            if "(+" in d:    st.success(f"✅ {d}")
            elif "(-" in d:  st.error(f"❌ {d}")
            else:             st.info(f"ℹ️ {d}")

    with col_ml:
        st.subheader(f"🤖 ML Forecast Table — D+5 to D+{forecast_days}")
        if not ml_df.empty and symbol in ml_df["symbol"].values:
            ml_row     = ml_df[ml_df["symbol"] == symbol].iloc[0]
            last_close = float(sig.get("close", 0))

            checkpoints = list(range(5, forecast_days + 1, 5))
            if forecast_days not in checkpoints:
                checkpoints.append(forecast_days)

            rows = []
            for d in checkpoints:
                row = {"Day": f"D+{d}  (W{d//5})"}
                model_vals = []
                for pfx, lbl in [("arima","ARIMA"),("prophet","Prophet"),("xgb","XGBoost")]:
                    v = ml_row.get(f"{pfx}_d{d}", np.nan)
                    if not pd.isna(v):
                        chg = (float(v) - last_close) / last_close * 100
                        row[lbl] = f"₹{float(v):,.0f} ({chg:+.1f}%)"
                        model_vals.append(float(v))
                    else:
                        row[lbl] = "—"
                if model_vals:
                    med = np.median(model_vals)
                    chg = (med - last_close) / last_close * 100
                    row["Consensus"] = f"₹{med:,.0f} ({chg:+.1f}%)"
                rows.append(row)

            st.dataframe(
                pd.DataFrame(rows).set_index("Day"),
                use_container_width=True,
                height=min(38 * len(rows) + 40, 520),
            )
            st.caption(
                "Weekly milestones · "
                "Consensus = median of ARIMA + Prophet + XGBoost"
            )
        else:
            st.warning("No ML data. Commit `02_data/ml_forecasts.csv` to repo.")


# ==============================================================================
# BACKTEST RESULTS
# ==============================================================================

if not bt_df.empty:
    st.markdown("---")
    st.subheader("📊 Backtest Results — last 6 months · your exact trading rules")

    total    = len(bt_df)
    wins     = (bt_df["outcome"] == "WIN").sum()
    losses   = (bt_df["outcome"] == "LOSS").sum()
    win_rate = wins / total * 100 if total else 0
    tot_pnl  = bt_df["pnl"].sum()
    avg_pnl  = bt_df["pnl"].mean()

    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("BUY Signals",    total)
    b2.metric("Win Rate",       f"{win_rate:.1f}%")
    b3.metric("Wins / Losses",  f"{wins} / {losses}")
    b4.metric("Timeouts",       total - wins - losses,
              help="Held full period, neither TP nor SL hit")
    b5.metric("Total P&L",      f"₹{tot_pnl:,.0f}",
              delta=f"₹{avg_pnl:,.0f} avg")

    sym_bt = bt_df[bt_df["symbol"] == symbol]
    if not sym_bt.empty:
        st.subheader(f"📌 {symbol} — Trade History")
        cols = [c for c in ["date","signal","score","entry","sl","tp",
                             "outcome","days_held","qty","pnl"] if c in sym_bt.columns]
        st.dataframe(sym_bt[cols].tail(20), use_container_width=True)

    st.subheader("📊 Backtest P&L — All 50 Stocks")
    pnl_sym = bt_df.groupby("symbol")["pnl"].sum().sort_values()
    fig_bt  = go.Figure(go.Bar(
        x=pnl_sym.values, y=pnl_sym.index, orientation="h",
        marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in pnl_sym.values],
        text=[f"₹{v:,.0f}" for v in pnl_sym.values], textposition="outside",
    ))
    fig_bt.update_layout(
        height=600, template="plotly_dark",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        margin=dict(l=130, r=80, t=10, b=0),
        xaxis_title="Total P&L (₹)",
    )
    st.plotly_chart(fig_bt, use_container_width=True)


# ==============================================================================
# ALL 50 SIGNALS TABLE
# ==============================================================================

st.markdown("---")
st.subheader("📋 All 50 Stocks — Current Signals")

if not signals_df.empty:
    preferred = ["symbol","close","signal","score","rsi","rsi_zone",
                 "cross_type","long_term_trend","ml_consensus_d1","obv_rising"]
    show_cols = [c for c in preferred if c in signals_df.columns]

    def color_signal(val):
        return {
            "STRONG BUY":   "background-color:#003300;color:#00ff88",
            "BUY":          "background-color:#002200;color:#88ff44",
            "NEUTRAL/HOLD": "background-color:#332200;color:#ffcc00",
            "SELL":         "background-color:#330011;color:#ff6644",
            "STRONG SELL":  "background-color:#220000;color:#ff0044",
        }.get(val, "")

    fmt = {}
    if "close"           in show_cols: fmt["close"]           = "₹{:,.2f}"
    if "score"           in show_cols: fmt["score"]           = "{:+d}"
    if "rsi"             in show_cols: fmt["rsi"]             = "{:.1f}"
    if "ml_consensus_d1" in show_cols: fmt["ml_consensus_d1"] = "₹{:,.2f}"

    st.dataframe(
        signals_df[show_cols].style.map(color_signal, subset=["signal"]).format(fmt, na_rep="—"),
        use_container_width=True, height=620,
    )
else:
    st.info(
        "Signal table appears once `02_data/latest_signals_summary.csv` "
        "is committed to your GitHub repo."
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#555;font-size:0.82rem;line-height:2;'>
<b>MBA Dissertation</b> · AI-Driven Algorithmic Trading System · SVNIT Surat<br>
NIFTY 50 · Equity Delivery Swing Trading · Capital ₹1,00,000 · RR 1:2 · No Shorting<br>
ML: ARIMA + Prophet + XGBoost · Consensus = Median · Indicators: RSI · SMA 20/50/200 · OBV · BB<br>
<span style='color:#333'>⚠ For dissertation / educational purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)
