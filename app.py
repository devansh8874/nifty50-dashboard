# ==============================================================================
#  NIFTY 50 ALGO TRADING DASHBOARD — Streamlit App  [v5 — Professional]
#  MBA Dissertation: AI-Driven Algorithmic Trading System — SVNIT Surat
# ------------------------------------------------------------------------------
#  CHANGES FROM v4:
#    - NEW: Interactive candlestick chart with click-to-zoom
#    - NEW: MACD subplot with histogram
#    - NEW: ADX subplot with +DI/-DI
#    - NEW: EMA 40 disaster line on chart
#    - NEW: XGBoost Classifier confidence display
#    - NEW: ML-enhanced signal (strategy + ML agreement)
#    - NEW: Confirmation checklist visual
#    - NEW: Trailing stop-loss display
#    - NEW: Backtest breakdown by outcome type
#    - IMPROVED: Better chart aesthetics + annotations
#    - MODELS: LSTM + XGBoost Classifier + SimpleRNN
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ── All 50 NIFTY symbols ─────────────────────────────────────────────────────
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
    page_title="NIFTY 50 | AI Trading Dashboard v5",
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
.confirm-yes { color:#00ff88; font-weight:bold; }
.confirm-no  { color:#ff4444; font-weight:bold; }
.signal-box {
    border-radius:8px; padding:16px; text-align:center; font-size:1.2rem;
    font-weight:bold; margin-bottom:10px;
}
.signal-buy    { background:#003300; color:#00ff88; border:2px solid #00ff88; }
.signal-sell   { background:#330000; color:#ff4444; border:2px solid #ff4444; }
.signal-hold   { background:#332200; color:#ffcc00; border:2px solid #ffcc00; }
.signal-exit   { background:#440000; color:#ff0000; border:2px solid #ff0000; }
</style>""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING
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

st.sidebar.title("📊 NIFTY 50 AI Dashboard v5")

if not signals_df.empty:
    as_of = signals_df["as_of_date"].iloc[0] if "as_of_date" in signals_df.columns else "—"
    st.sidebar.success(f"✅ Signals as of: **{as_of}**")
    if last_run:
        st.sidebar.caption(f"Pipeline ran: {last_run}")
else:
    st.sidebar.error("⚠️ No data. Run pipeline first.")

st.sidebar.markdown("---")

# Stock selector
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
candles = st.sidebar.slider("Historical candles", 60, 500, 200, step=20)
forecast_days = st.sidebar.select_slider(
    "ML Forecast horizon",
    options=[5, 21, 42, 63, 84],
    value=84,
    format_func=lambda x: {
        5:"1 week", 21:"1 month", 42:"2 months",
        63:"3 months", 84:"4 months",
    }[x],
)
st.sidebar.markdown("---")
show_forecast = st.sidebar.checkbox("📈 ML forecast lines", value=True)
show_bb       = st.sidebar.checkbox("📊 Bollinger Bands",   value=True)
show_sma200   = st.sidebar.checkbox("📉 SMA 200",           value=True)
show_ema40    = st.sidebar.checkbox("🚨 EMA 40 (disaster)",  value=True)
show_macd     = st.sidebar.checkbox("📊 MACD subplot",       value=True)
show_adx      = st.sidebar.checkbox("📊 ADX subplot",        value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Your Strategy")
st.sidebar.markdown("""
**PRIMARY (50%):** 20/50 SMA crossover  
**CONFIRMATION (50%):** ALL must be ✅  
1. RSI > 50 (or rising)  
2. MACD bullish  
3. BB not overbought  
4. Price > EMA 40  
5. Price > SMA 200  
6. OBV rising  
7. ADX > 25 (bullish)  
8. SMA 50 > SMA 200 (regime)  

**EXIT:** Price < EMA 40 | Death cross  
**Trailing SL:** 4%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Ensemble v5")
st.sidebar.markdown("""
- **LSTM** — 64→32 units, 60d seq  
- **XGBoost Clf** — binary ≥2% in 10d  
- **SimpleRNN** — 48→24 units  
- **SHAP** — feature selection  
- **Consensus** — median + majority vote
""")


# ==============================================================================
# HEADER
# ==============================================================================

c_title, c_btn = st.columns([5, 1])
with c_title:
    st.title(f"📈 {symbol} — AI Trading Dashboard")
with c_btn:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Data freshness
if not ind_df.empty and "date" in ind_df.columns:
    max_date = pd.to_datetime(ind_df["date"]).max()
    days_old = (pd.Timestamp.now() - max_date).days
    if days_old <= 2:
        st.markdown(
            f'<div class="data-banner">📅 Data as of <b>{max_date.date()}</b> · '
            f'ML: LSTM + XGBoost Classifier + SimpleRNN</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"⚠️ Data is {days_old} days old (last: {max_date.date()}).")
else:
    st.error("📂 No data found. Run the pipeline first.")

st.markdown("---")


# ==============================================================================
# SIGNAL METRICS + CONFIRMATION CHECKLIST
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]

    # Signal box
    signal_label = str(sig.get("signal", "HOLD"))
    signal_class = {
        "STRONG BUY": "signal-buy", "BUY": "signal-buy",
        "SELL": "signal-sell", "EXIT": "signal-exit",
    }.get(signal_label, "signal-hold")
    ml_enhanced = str(sig.get("ml_enhanced_signal", signal_label))

    col_sig, col_conf = st.columns([1, 2])

    with col_sig:
        st.markdown(f'<div class="signal-box {signal_class}">{signal_label}<br>'
                    f'<span style="font-size:0.8rem">Score: {sig.get("score", 0):+d} · '
                    f'{sig.get("confirmations", "?/?")} confirmations</span></div>',
                    unsafe_allow_html=True)

        # ML-enhanced signal
        if ml_enhanced != signal_label:
            st.info(f"🤖 **ML Enhanced:** {ml_enhanced}")

        # Key metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"₹{float(sig.get('close', 0)):,.2f}")
        c2.metric("RSI",   f"{float(sig.get('rsi', 50)):.1f}")
        adx_val = sig.get("adx", 0)
        c3.metric("ADX",   f"{float(adx_val):.1f}" if adx_val else "—")

    with col_conf:
        st.markdown("### ✅ Confirmation Checklist")
        details_str = str(sig.get("signal_details", ""))

        # Parse confirmation details
        for line in details_str.split(" | "):
            line = line.strip()
            if not line:
                continue
            if "✅" in line or "(+" in line:
                st.success(line)
            elif "❌" in line or "(-" in line:
                st.error(line)
            elif "⚠" in line:
                st.warning(line)
            elif "🚨" in line:
                st.error(line)
            elif "★" in line:
                st.info(line)
            elif "🟢" in line or "🟡" in line:
                st.success(line)
            elif "🔴" in line:
                st.error(line)
            else:
                st.info(line)


# ==============================================================================
# TRADE PLAN
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]
    if str(sig.get("signal", "")) in ("BUY", "STRONG BUY"):
        st.markdown("---")
        st.subheader("💰 Trade Plan")
        tp1, tp2, tp3, tp4, tp5, tp6 = st.columns(6)
        tp1.metric("Entry",       f"₹{sig.get('entry', 0):,}")
        tp2.metric("Stop Loss",   f"₹{sig.get('stop_loss', 0):,}  ({sig.get('sl_pct','')})")
        tp3.metric("Target",      f"₹{sig.get('target', 0):,}  ({sig.get('upside_pct','')})")
        tp4.metric("Trailing SL", str(sig.get("trailing_stop", "4%")))
        tp5.metric("Qty",         str(sig.get("qty", "N/A")))
        tp6.metric("Position",    str(sig.get("position_value", "N/A")))

        st.info(f"📅 **Hold**: {sig.get('hold_duration', 'N/A')}")
        st.caption("Exit rules: TP hit | Trailing SL hit | Price < EMA 40 | Death cross")

        # XGBoost classifier confidence
        xgb_proba = sig.get("xgb_proba", None)
        if xgb_proba and not pd.isna(xgb_proba):
            xgb_signal = sig.get("xgb_signal", "")
            if xgb_signal == "BULLISH":
                st.success(f"🤖 **XGBoost ML**: {float(xgb_proba):.0%} probability of ≥2% rise in 10 days")
            else:
                st.warning(f"🤖 **XGBoost ML**: Only {float(xgb_proba):.0%} probability — ML is cautious")


# ==============================================================================
# MAIN CHART — Interactive Candlestick + All Indicators
# ==============================================================================

st.markdown("---")
label_map = {5:"1 wk", 21:"1 mo", 42:"2 mo", 63:"3 mo", 84:"4 mo"}
st.subheader(
    f"📊 {symbol} — Interactive Chart + {label_map.get(forecast_days,'')} ML Forecast"
)

# Count subplots
n_subplots = 3  # price + RSI + OBV (always)
subplot_titles = [f"{symbol} — Price + Indicators", "RSI (14)", "OBV"]
row_heights = [0.50, 0.15, 0.15]

if show_macd:
    n_subplots += 1
    subplot_titles.append("MACD")
    row_heights.append(0.10)
if show_adx:
    n_subplots += 1
    subplot_titles.append("ADX")
    row_heights.append(0.10)

# Normalize heights
total_h = sum(row_heights)
row_heights = [h/total_h for h in row_heights]

if not ind_df.empty and symbol in ind_df["symbol"].values:
    s_df = ind_df[ind_df["symbol"] == symbol].copy()
    s_df["date"] = pd.to_datetime(s_df["date"])
    s_df = s_df.sort_values("date").tail(candles).copy()

    fig = make_subplots(
        rows=n_subplots, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Row 1: Candlestick ──────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=s_df["date"], open=s_df["open"], high=s_df["high"],
        low=s_df["low"], close=s_df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # SMAs
    for col_name, color, width, dash in [
        ("MA_20", "#ff9800", 1.5, None),
        ("MA_50", "#2196f3", 1.5, None),
    ]:
        if col_name in s_df.columns:
            fig.add_trace(go.Scatter(
                x=s_df["date"], y=s_df[col_name],
                name=col_name.replace("_"," "),
                line=dict(color=color, width=width, dash=dash),
            ), row=1, col=1)

    if show_sma200 and "MA_200" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["MA_200"],
            name="SMA 200", line=dict(color="#ff4444", width=1.5, dash="dot"),
        ), row=1, col=1)

    # EMA 40 (disaster line)
    if show_ema40 and "EMA_40" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["EMA_40"],
            name="EMA 40 (disaster)",
            line=dict(color="#ff00ff", width=2, dash="dash"),
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

    # ── ML Forecast Lines ───────────────────────────────────────────────
    if show_forecast and not ml_df.empty and symbol in ml_df["symbol"].values:
        ml_row       = ml_df[ml_df["symbol"] == symbol].iloc[0]
        last_date    = s_df["date"].max()
        last_close   = float(s_df["close"].iloc[-1])
        future_dates = pd.bdate_range(last_date, periods=forecast_days + 1)[1:]

        model_cfg = [
            ("lstm", "#9c27b0", "dot",     "LSTM"),
            ("rnn",  "#00bcd4", "dashdot", "SimpleRNN"),
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

        # Consensus line
        if len(all_vals) >= 2:
            consensus = []
            for i in range(forecast_days):
                day_v = [v[i] for v in all_vals.values()
                         if i < len(v) and not pd.isna(v[i])]
                consensus.append(float(np.median(day_v)) if day_v else np.nan)

            valid_c = [(future_dates[i], v) for i, v in enumerate(consensus)
                       if i < len(future_dates) and not np.isnan(v)]
            if valid_c:
                xc = [last_date] + [p[0] for p in valid_c]
                yc = [last_close] + [p[1] for p in valid_c]

                # ±5% uncertainty band
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

                end_chg = (yc[-1] - last_close) / last_close * 100
                fig.add_annotation(
                    x=xc[-1], y=yc[-1],
                    text=f"  ₹{yc[-1]:,.0f} ({end_chg:+.1f}%)",
                    showarrow=False, font=dict(color="#ffffff", size=11),
                    xanchor="left",
                )

    # ── Row 2: RSI ──────────────────────────────────────────────────────
    if "RSI" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["RSI"],
            name="RSI", line=dict(color="#e91e63", width=1.5),
        ), row=2, col=1)
        for lvl, clr in [(70, "rgba(255,100,100,0.4)"), (50, "rgba(255,255,255,0.2)"),
                         (30, "rgba(100,255,100,0.4)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr, row=2, col=1)

    # ── Row 3: OBV ──────────────────────────────────────────────────────
    if "OBV" in s_df.columns:
        obv_c = ["#26a69a" if r else "#ef5350"
                 for r in s_df["OBV_Rising"].fillna(False)]
        fig.add_trace(go.Bar(
            x=s_df["date"], y=s_df["OBV"], name="OBV", marker_color=obv_c,
        ), row=3, col=1)

    # ── Row 4: MACD (optional) ──────────────────────────────────────────
    current_row = 4
    if show_macd and "MACD" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["MACD"],
            name="MACD", line=dict(color="#2196f3", width=1.5),
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["MACD_Signal"],
            name="Signal", line=dict(color="#ff9800", width=1.5),
        ), row=current_row, col=1)
        if "MACD_Hist" in s_df.columns:
            macd_colors = ["#26a69a" if v >= 0 else "#ef5350"
                          for v in s_df["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=s_df["date"], y=s_df["MACD_Hist"],
                name="MACD Hist", marker_color=macd_colors,
            ), row=current_row, col=1)
        current_row += 1

    # ── Row 5: ADX (optional) ──────────────────────────────────────────
    if show_adx and "ADX" in s_df.columns:
        fig.add_trace(go.Scatter(
            x=s_df["date"], y=s_df["ADX"],
            name="ADX", line=dict(color="#ffffff", width=2),
        ), row=current_row, col=1)
        if "Plus_DI" in s_df.columns:
            fig.add_trace(go.Scatter(
                x=s_df["date"], y=s_df["Plus_DI"],
                name="+DI", line=dict(color="#26a69a", width=1),
            ), row=current_row, col=1)
        if "Minus_DI" in s_df.columns:
            fig.add_trace(go.Scatter(
                x=s_df["date"], y=s_df["Minus_DI"],
                name="-DI", line=dict(color="#ef5350", width=1),
            ), row=current_row, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="rgba(255,255,0,0.5)",
                      row=current_row, col=1)

    # ── Layout ──────────────────────────────────────────────────────────
    chart_height = 700 + (show_macd * 120) + (show_adx * 120)
    fig.update_layout(
        height=chart_height, template="plotly_dark",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_yaxes(gridcolor="#1e1e2e")
    fig.update_xaxes(gridcolor="#1e1e2e")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

else:
    st.warning(f"No chart data for **{symbol}**.")


# ==============================================================================
# ML FORECAST TABLE
# ==============================================================================

if not signals_df.empty and symbol in signals_df["symbol"].values:
    sig = signals_df[signals_df["symbol"] == symbol].iloc[0]
    st.markdown("---")
    col_ml, col_xgb = st.columns(2)

    with col_ml:
        st.subheader(f"🤖 ML Forecast — D+5 to D+{forecast_days}")
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
                for pfx, lbl in [("lstm", "LSTM"), ("rnn", "RNN")]:
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
            st.caption("Consensus = median of LSTM + SimpleRNN")
        else:
            st.warning("No ML forecast data available.")

    with col_xgb:
        st.subheader("🎯 XGBoost Classifier")
        xgb_pred  = sig.get("xgb_pred", None)
        xgb_proba = sig.get("xgb_proba", None)
        xgb_sig   = sig.get("xgb_signal", None)

        if xgb_proba and not pd.isna(xgb_proba):
            proba_val = float(xgb_proba)
            st.metric("Prediction", f"{'🟢 BULLISH' if xgb_sig=='BULLISH' else '🔴 BEARISH'}")
            st.metric("Confidence", f"{proba_val:.0%}")
            st.progress(proba_val)
            st.caption(f"Binary: will {symbol} rise ≥2% in next 10 trading days?")

            # Model agreement
            strategy_signal = str(sig.get("signal", "HOLD"))
            if strategy_signal in ("BUY", "STRONG BUY") and xgb_sig == "BULLISH":
                st.success("✅ **Strategy + ML agree**: strong conviction")
            elif strategy_signal in ("BUY", "STRONG BUY") and xgb_sig != "BULLISH":
                st.warning("⚠️ **Strategy bullish, ML cautious**: reduce position size")
            elif strategy_signal not in ("BUY", "STRONG BUY") and xgb_sig == "BULLISH":
                st.info("🔍 **ML bullish, strategy waiting**: watch for crossover")
        else:
            st.info("XGBoost classifier data not available for this stock.")


# ==============================================================================
# BACKTEST RESULTS
# ==============================================================================

if not bt_df.empty:
    st.markdown("---")
    st.subheader("📊 Backtest Results — Your Strategy + Trailing SL")

    total     = len(bt_df)
    wins      = bt_df["outcome"].str.contains("WIN", na=False).sum()
    losses    = bt_df["outcome"].str.contains("LOSS", na=False).sum()
    exits     = bt_df["outcome"].str.contains("EXIT", na=False).sum()
    timeouts  = total - wins - losses - exits
    win_rate  = wins / total * 100 if total else 0
    tot_pnl   = bt_df["pnl"].sum()
    avg_pnl   = bt_df["pnl"].mean()

    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Total Trades",  total)
    b2.metric("Win Rate",      f"{win_rate:.1f}%")
    b3.metric("Wins",          wins)
    b4.metric("Losses",        losses)
    b5.metric("Exits",         exits, help="EMA40 breach or death cross exits")
    b6.metric("Total P&L",     f"₹{tot_pnl:,.0f}", delta=f"₹{avg_pnl:,.0f} avg")

    # Stock-specific backtest
    sym_bt = bt_df[bt_df["symbol"] == symbol]
    if not sym_bt.empty:
        st.subheader(f"📌 {symbol} — Trade History")
        cols = [c for c in ["date","signal","confirmations","entry","exit_price",
                            "sl","tp","outcome","days_held","pnl","pnl_pct"]
                if c in sym_bt.columns]
        st.dataframe(sym_bt[cols].tail(20), use_container_width=True)

    # P&L by stock
    st.subheader("📊 P&L by Stock")
    pnl_sym = bt_df.groupby("symbol")["pnl"].sum().sort_values()
    fig_bt  = go.Figure(go.Bar(
        x=pnl_sym.values, y=pnl_sym.index, orientation="h",
        marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in pnl_sym.values],
        text=[f"₹{v:,.0f}" for v in pnl_sym.values], textposition="outside",
    ))
    fig_bt.update_layout(
        height=max(400, len(pnl_sym) * 18),
        template="plotly_dark",
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
    preferred = ["symbol", "close", "signal", "score", "confirmations",
                 "rsi", "adx", "macd_bullish", "disaster_line",
                 "trend_regime", "obv_rising", "xgb_signal", "xgb_proba"]
    show_cols = [c for c in preferred if c in signals_df.columns]

    def color_signal(val):
        return {
            "STRONG BUY":   "background-color:#003300;color:#00ff88",
            "BUY":          "background-color:#002200;color:#88ff44",
            "HOLD":         "background-color:#332200;color:#ffcc00",
            "SELL":         "background-color:#330011;color:#ff6644",
            "EXIT":         "background-color:#220000;color:#ff0044",
        }.get(val, "")

    fmt = {}
    if "close"     in show_cols: fmt["close"]     = "₹{:,.2f}"
    if "score"     in show_cols: fmt["score"]      = "{:+d}"
    if "rsi"       in show_cols: fmt["rsi"]        = "{:.1f}"
    if "adx"       in show_cols: fmt["adx"]        = "{:.1f}"
    if "xgb_proba" in show_cols: fmt["xgb_proba"]  = "{:.0%}"

    styled = signals_df[show_cols].style
    if "signal" in show_cols:
        styled = styled.map(color_signal, subset=["signal"])
    styled = styled.format(fmt, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=620)
else:
    st.info("Signal table appears once pipeline data is available.")


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#555;font-size:0.82rem;line-height:2;'>
<b>MBA Dissertation v5</b> · AI-Driven Algorithmic Trading System · SVNIT Surat<br>
NIFTY 50 · Equity Delivery Swing Trading · Capital ₹1,00,000 · RR 1:2 · No Shorting<br>
Strategy: 20/50 SMA crossover + 7 confirmations (RSI, MACD, BB, EMA40, MA200, OBV, ADX)<br>
ML: LSTM + XGBoost Classifier + SimpleRNN · SHAP Feature Selection · Trailing SL 4%<br>
<span style='color:#333'>⚠ For dissertation / educational purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)
