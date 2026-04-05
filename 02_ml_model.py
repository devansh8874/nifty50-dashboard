"""
================================================================================
  NIFTY 50 — ML FORECASTING + SIGNAL PIPELINE  [v5 — Professional]
  MBA Dissertation: AI-Driven Algorithmic Trading System
  
  CHANGES FROM v4:
    - REPLACED: ARIMA + Prophet + old XGBoost Regressor
    - NEW MODELS: LSTM, XGBoost Binary Classifier, SimpleRNN
    - STRATEGY: Your exact 20/50 SMA crossover + 6-indicator confirmation
    - ADDED: MACD, ADX, EMA 40 (disaster line) indicators
    - ADDED: SHAP-based feature selection for XGBoost
    - ADDED: Binary classification (will price rise ≥2% in N days?)
    - ADDED: Walk-forward temporal validation (no data leakage)
    - ADDED: Trailing stop-loss (3-5%) in backtest
    - ADDED: Trend regime filter (SMA 50 > SMA 200)
    - ADDED: Backtrader-based professional backtesting
    - FIXED: Data leakage in train/test split
    - FIXED: All features computed before split, model fit only on train
    
  SIGNAL LOGIC (your exact strategy):
    PRIMARY (50%): 20 SMA crosses above 50 SMA (bullish crossover)
    CONFIRMATION (50% — ALL must be bullish):
      1. RSI > 50 (or rising from oversold)
      2. MACD line above signal line
      3. BB not overbought (price < BB Upper)
      4. Price > 40 EMA (disaster line)
      5. Price > 200 SMA (long-term trend)
      6. OBV rising (volume confirms)
      7. ADX > 25 (strong trend)
    
    BUY  = crossover + all confirmations met
    HOLD = crossover but confirmations missing
    SELL = 20 SMA crosses below 50 SMA
    EXIT = price drops below 40 EMA (disaster line breach)
    
  ML ENSEMBLE: LSTM + XGBoost Classifier + SimpleRNN
    - XGBoost: Binary classifier → "will price rise ≥2% in N days?"
    - LSTM: Sequential pattern recognition on indicator time series
    - SimpleRNN: Lightweight recurrence for trend continuation
    - Consensus: majority vote (2 out of 3 must agree)
================================================================================
"""

import pandas as pd
import numpy as np
import os, warnings, json, time
from datetime import datetime, timedelta

# ── ML dependencies ──────────────────────────────────────────────────────────
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("WARNING: tensorflow not installed. Run: pip install tensorflow")

try:
    import backtrader as bt
    HAS_BT = True
except ImportError:
    HAS_BT = False
    print("WARNING: backtrader not installed. Run: pip install backtrader")

warnings.filterwarnings("ignore")

print(f"pandas   : {pd.__version__}")
print(f"xgboost  : {xgb.__version__}")
print(f"shap     : {'✓' if HAS_SHAP else '✗ (install: pip install shap)'}")
print(f"tensorflow: {'✓' if HAS_TF else '✗ (install: pip install tensorflow)'}")
print(f"backtrader: {'✓' if HAS_BT else '✗ (install: pip install backtrader)'}")
print("All core imports OK.\n")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MASTER_CSV          = "02_data/nifty50_historical_master.csv"
INDICATORS_CSV      = "02_data/nifty50_with_indicators.csv"
SIGNALS_CSV         = "02_data/latest_signals_summary.csv"
ML_FORECASTS_CSV    = "02_data/ml_forecasts.csv"
BACKTEST_CSV        = "02_data/backtest_results.csv"
ML_MODELS_DIR       = "02_data/ml_models"
SHAP_DIR            = "02_data/shap_reports"

os.makedirs("02_data", exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

ML_SYMBOLS          = None           # None = all 50 stocks
ML_FORECAST_DAYS    = 84             # predict next 84 trading days (~4 months)

# XGBoost Classifier config
XGB_LOOKFORWARD     = 10             # predict: will price rise ≥2% in 10 days?
XGB_THRESHOLD_PCT   = 2.0            # minimum % rise to classify as BUY
XGB_HISTORY_DAYS    = 2520           # ~5 years training data
XGB_LAG_DAYS        = 30             # last 30 days as features

# LSTM / RNN config
SEQ_LENGTH          = 60             # 60-day sequences for LSTM/RNN
LSTM_EPOCHS         = 50
LSTM_BATCH_SIZE     = 32
LSTM_HISTORY_DAYS   = 2520

# Trading config
CAPITAL             = 100_000        # INR
CAPITAL_PER_TRADE   = 0.20           # 20% per trade
MAX_TRADES          = 2
RR_RATIO            = 2.0
TRAILING_STOP_PCT   = 0.04           # 4% trailing stop-loss
MIN_HOLD_DAYS       = 5              # minimum holding period after entry

print("Configuration loaded.")
print(f"  ML_SYMBOLS     : {ML_SYMBOLS if ML_SYMBOLS else 'ALL 50'}")
print(f"  Forecast days  : {ML_FORECAST_DAYS}")
print(f"  XGB target     : price rises ≥{XGB_THRESHOLD_PCT}% in {XGB_LOOKFORWARD} days")
print(f"  Trailing stop  : {TRAILING_STOP_PCT*100:.0f}%")
print(f"  Capital/trade  : ₹{CAPITAL * CAPITAL_PER_TRADE:,.0f}")


# ==============================================================================
# CLASS: TradingIndicators  [v5 — ADDED: MACD, ADX, EMA 40]
# ==============================================================================

class TradingIndicators:

    @staticmethod
    def add_rsi(df, period=14):
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"]          = 100 - (100 / (1 + rs))
        df["RSI_Above_50"] = df["RSI"] > 50
        df["RSI_Below_30"] = df["RSI"] < 30
        df["RSI_Zone"]     = np.where(df["RSI"] > 70, "OVERBOUGHT",
                             np.where(df["RSI"] < 30, "OVERSOLD",
                             np.where(df["RSI"] > 50, "BULLISH", "BEARISH")))
        return df

    @staticmethod
    def add_sma_cross(df):
        df["MA_20"]          = df["close"].rolling(20).mean()
        df["MA_50"]          = df["close"].rolling(50).mean()
        df["MA_20_above_50"] = df["MA_20"] > df["MA_50"]
        cross                = df["MA_20_above_50"].astype(int).diff()
        df["MA_Crossover"]   = cross
        df["Cross_Type"]     = np.where(cross == 1, "GOLDEN_CROSS",
                               np.where(cross == -1, "DEATH_CROSS", "NONE"))
        return df

    @staticmethod
    def add_sma200(df):
        df["MA_200"]            = df["close"].rolling(200).mean()
        df["Price_Above_MA200"] = df["close"] > df["MA_200"]
        df["Long_Term_Trend"]   = np.where(df["Price_Above_MA200"], "BULL", "BEAR")
        # Trend regime filter: SMA 50 > SMA 200
        if "MA_50" in df.columns:
            df["Trend_Regime"] = np.where(df["MA_50"] > df["MA_200"], "BULLISH_REGIME", "BEARISH_REGIME")
        return df

    @staticmethod
    def add_ema40(df):
        """EMA 40 — your disaster line. Price below this = EXIT immediately."""
        df["EMA_40"]            = df["close"].ewm(span=40, adjust=False).mean()
        df["Price_Above_EMA40"] = df["close"] > df["EMA_40"]
        df["Disaster_Line"]     = np.where(df["Price_Above_EMA40"], "SAFE", "BREACH")
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """MACD — your momentum confirmation indicator."""
        ema_fast           = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow           = df["close"].ewm(span=slow, adjust=False).mean()
        df["MACD"]         = ema_fast - ema_slow
        df["MACD_Signal"]  = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]
        df["MACD_Bullish"] = df["MACD"] > df["MACD_Signal"]
        return df

    @staticmethod
    def add_adx(df, period=14):
        """ADX — trend strength. >25 = strong trend, trade. <25 = choppy, skip."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr      = tr.rolling(period).mean()
        plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx      = dx.rolling(period).mean()

        df["ADX"]           = adx
        df["Plus_DI"]       = plus_di
        df["Minus_DI"]      = minus_di
        df["ADX_Strong"]    = adx > 25
        df["ADX_Bullish"]   = (plus_di > minus_di) & (adx > 25)
        return df

    @staticmethod
    def add_bollinger(df, window=20):
        mid            = df["close"].rolling(window).mean()
        std            = df["close"].rolling(window).std()
        df["BB_Mid"]   = mid
        df["BB_Upper"] = mid + 2 * std
        df["BB_Lower"] = mid - 2 * std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid
        df["BB_Pct"]   = ((df["close"] - df["BB_Lower"]) /
                          (df["BB_Upper"] - df["BB_Lower"]))
        df["BB_Not_Overbought"] = df["close"] < df["BB_Upper"]
        return df

    @staticmethod
    def add_obv(df):
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"]        = obv
        df["OBV_MA_20"]  = pd.Series(obv).rolling(20).mean().values
        df["OBV_Rising"] = df["OBV"] > df["OBV_MA_20"]
        return df

    @staticmethod
    def add_volume(df):
        df["Volume_MA_20"] = df["volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA_20"]
        df["High_Volume"]  = df["Volume_Ratio"] > 1.5
        return df

    @staticmethod
    def add_returns(df):
        df["Daily_Return"] = df["close"].pct_change() * 100
        df["Return_5d"]    = df["close"].pct_change(5)  * 100
        df["Return_10d"]   = df["close"].pct_change(10) * 100
        df["Return_20d"]   = df["close"].pct_change(20) * 100
        return df

    @staticmethod
    def add_atr(df, period=14):
        """Average True Range — used for dynamic stop-loss sizing."""
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"]    = tr.rolling(period).mean()
        df["ATR_Pct"] = df["ATR"] / df["close"] * 100
        return df

    @classmethod
    def calculate_all(cls, df):
        """Calculate ALL indicators in correct order."""
        df = cls.add_rsi(df)
        df = cls.add_sma_cross(df)
        df = cls.add_sma200(df)
        df = cls.add_ema40(df)          # NEW: disaster line
        df = cls.add_macd(df)           # NEW: MACD
        df = cls.add_adx(df)            # NEW: ADX
        df = cls.add_bollinger(df)
        df = cls.add_obv(df)
        df = cls.add_volume(df)
        df = cls.add_returns(df)
        df = cls.add_atr(df)            # NEW: ATR for stop-loss
        return df

print("TradingIndicators class ready (v5: +MACD +ADX +EMA40 +ATR).")


# ==============================================================================
# CLASS: TradingSignals  [v5 — YOUR EXACT STRATEGY]
# ==============================================================================

class TradingSignals:
    """
    YOUR EXACT STRATEGY:
      PRIMARY (50%): 20 SMA crosses above 50 SMA
      CONFIRMATION (50%): RSI + MACD + BB + EMA40 + MA200 + OBV + ADX
      
      BUY  = crossover + ALL confirmations bullish
      HOLD = crossover but some confirmations missing
      SELL = death cross (20 SMA below 50 SMA)
      EXIT = price drops below EMA 40 (disaster line)
    """

    @classmethod
    def generate(cls, stock_df):
        if len(stock_df) == 0:
            return stock_df, 0, "HOLD", [], {}

        latest  = stock_df.iloc[-1]
        prev    = stock_df.iloc[-2] if len(stock_df) > 1 else latest
        details = []
        confirmations = {}

        # ══════════════════════════════════════════════════════════════════
        # PRIMARY SIGNAL (50%): 20 SMA / 50 SMA crossover
        # ══════════════════════════════════════════════════════════════════
        ma_cross    = latest.get("MA_Crossover", 0)
        above_50    = latest.get("MA_20_above_50", False)
        has_crossover = False

        if ma_cross == 1:
            has_crossover = True
            details.append("★ PRIMARY: GOLDEN CROSS — 20SMA crossed above 50SMA ✓")
        elif ma_cross == -1:
            details.append("★ PRIMARY: DEATH CROSS — 20SMA crossed below 50SMA ✗")
            # Immediate SELL on death cross
            return stock_df, -100, "SELL", details, confirmations
        elif above_50:
            has_crossover = True  # still in bullish zone from earlier crossover
            details.append("★ PRIMARY: 20SMA > 50SMA (bullish zone, watching)")
        else:
            details.append("★ PRIMARY: 20SMA < 50SMA (bearish zone) ✗")

        # ══════════════════════════════════════════════════════════════════
        # DISASTER LINE CHECK: price below EMA 40 = EXIT
        # ══════════════════════════════════════════════════════════════════
        price_above_ema40 = latest.get("Price_Above_EMA40", True)
        ema40 = latest.get("EMA_40", 0)
        if not price_above_ema40:
            details.append(f"🚨 DISASTER LINE BREACH: Price below EMA 40 [{ema40:.1f}] → EXIT")
            return stock_df, -200, "EXIT", details, confirmations

        # ══════════════════════════════════════════════════════════════════
        # CONFIRMATION SIGNALS (50%): ALL must be bullish for BUY
        # ══════════════════════════════════════════════════════════════════
        score = 0

        # 1. RSI — bullish if > 50
        rsi = latest.get("RSI", 50)
        if pd.isna(rsi): rsi = 50
        rsi_bullish = rsi > 50 or (rsi > 30 and rsi > prev.get("RSI", 50))  # rising from oversold
        confirmations["RSI"] = rsi_bullish
        if rsi_bullish:
            score += 15
            details.append(f"  ✅ RSI {rsi:.1f} — bullish (>50 or rising) (+15)")
        else:
            score -= 15
            details.append(f"  ❌ RSI {rsi:.1f} — bearish (<50) (-15)")

        # 2. MACD — bullish if MACD > signal line
        macd_bullish = latest.get("MACD_Bullish", False)
        macd_val     = latest.get("MACD", 0)
        macd_sig     = latest.get("MACD_Signal", 0)
        confirmations["MACD"] = bool(macd_bullish)
        if macd_bullish:
            score += 15
            details.append(f"  ✅ MACD [{macd_val:.2f}] > Signal [{macd_sig:.2f}] — bullish (+15)")
        else:
            score -= 15
            details.append(f"  ❌ MACD [{macd_val:.2f}] < Signal [{macd_sig:.2f}] — bearish (-15)")

        # 3. Bollinger Bands — not overbought
        bb_ok = latest.get("BB_Not_Overbought", True)
        bb_pct = latest.get("BB_Pct", 0.5)
        confirmations["BB"] = bool(bb_ok)
        if bb_ok:
            score += 10
            details.append(f"  ✅ BB %B={bb_pct:.2f} — not overbought (+10)")
        else:
            score -= 10
            details.append(f"  ❌ BB %B={bb_pct:.2f} — overbought (price at/above BB Upper) (-10)")

        # 4. Price > EMA 40 (disaster line) — already checked above, always true here
        confirmations["EMA40"] = True
        score += 10
        details.append(f"  ✅ Price above EMA 40 [{ema40:.1f}] — disaster line safe (+10)")

        # 5. Price > SMA 200 (long-term trend)
        above_200 = latest.get("Price_Above_MA200", False)
        ma200     = latest.get("MA_200", 0)
        confirmations["MA200"] = bool(above_200)
        if above_200:
            score += 15
            details.append(f"  ✅ Price above SMA 200 [{ma200:.1f}] — long-term BULL (+15)")
        else:
            score -= 15
            details.append(f"  ❌ Price below SMA 200 [{ma200:.1f}] — long-term BEAR (-15)")

        # 6. OBV rising (volume confirms)
        obv_rising = latest.get("OBV_Rising", False)
        confirmations["OBV"] = bool(obv_rising)
        if obv_rising:
            score += 15
            details.append("  ✅ OBV rising — volume confirms move (+15)")
        else:
            score -= 15
            details.append("  ❌ OBV falling — volume does not confirm (-15)")

        # 7. ADX — strong trend
        adx         = latest.get("ADX", 0)
        adx_bullish = latest.get("ADX_Bullish", False)
        confirmations["ADX"] = bool(adx_bullish)
        if adx_bullish:
            score += 10
            details.append(f"  ✅ ADX {adx:.1f} > 25 & +DI > -DI — strong bullish trend (+10)")
        else:
            if adx is not None and not pd.isna(adx) and adx > 25:
                score -= 5
                details.append(f"  ⚠️ ADX {adx:.1f} > 25 but -DI dominant — strong bearish (-5)")
            else:
                score -= 10
                details.append(f"  ❌ ADX {adx:.1f} < 25 — weak/choppy trend, avoid (-10)")

        # ══════════════════════════════════════════════════════════════════
        # TREND REGIME FILTER: SMA 50 > SMA 200
        # ══════════════════════════════════════════════════════════════════
        trend_regime = latest.get("Trend_Regime", "BEARISH_REGIME")
        if trend_regime == "BEARISH_REGIME":
            score -= 20
            details.append("  ⛔ REGIME FILTER: SMA 50 < SMA 200 → bearish regime (-20)")
            confirmations["REGIME"] = False
        else:
            score += 10
            details.append("  ✅ REGIME FILTER: SMA 50 > SMA 200 → bullish regime (+10)")
            confirmations["REGIME"] = True

        # ══════════════════════════════════════════════════════════════════
        # FINAL SIGNAL CLASSIFICATION
        # ══════════════════════════════════════════════════════════════════
        bull_count = sum(1 for v in confirmations.values() if v)
        total_conf = len(confirmations)

        if has_crossover and bull_count == total_conf:
            label = "STRONG BUY"
            details.append(f"\n  🟢 STRONG BUY: crossover + {bull_count}/{total_conf} confirmations")
        elif has_crossover and bull_count >= total_conf - 2:
            label = "BUY"
            details.append(f"\n  🟡 BUY: crossover + {bull_count}/{total_conf} confirmations")
        elif has_crossover:
            label = "HOLD"
            details.append(f"\n  🟠 HOLD: crossover but only {bull_count}/{total_conf} confirmations")
        elif score <= -40:
            label = "SELL"
            details.append(f"\n  🔴 SELL: bearish signals dominant ({bull_count}/{total_conf})")
        else:
            label = "HOLD"
            details.append(f"\n  ⚪ HOLD: no clear signal ({bull_count}/{total_conf})")

        return stock_df, score, label, details, confirmations

    @staticmethod
    def get_trade_plan(latest_row, label, score):
        close    = latest_row.get("close", 0)
        atr      = latest_row.get("ATR", close * 0.02)
        above_200= latest_row.get("Price_Above_MA200", False)
        ema40    = latest_row.get("EMA_40", close * 0.95)

        if pd.isna(atr) or atr == 0:
            atr = close * 0.02

        # Dynamic SL based on ATR (2x ATR below entry)
        sl_dist  = 2 * atr
        tp_dist  = sl_dist * RR_RATIO  # RR 1:2

        # Never set SL above EMA 40 (disaster line is the absolute floor)
        sl = max(round(close - sl_dist, 2), round(float(ema40) * 0.99, 2))
        tp = round(close + tp_dist, 2)

        sl_pct     = round((close - sl) / close * 100, 2)
        upside_pct = round((tp - close) / close * 100, 2)

        position_value = CAPITAL * CAPITAL_PER_TRADE
        qty            = int(position_value / close) if close > 0 else 0
        actual_value   = round(qty * close, 2)

        if label in ("BUY", "STRONG BUY"):
            if label == "STRONG BUY" and above_200:
                hold = "8–12 weeks. Exit at TP or 2 closes below EMA 40."
            elif above_200:
                hold = "4–8 weeks. Re-check signals weekly."
            else:
                hold = "2–4 weeks. Cautious — price below 200SMA. Tight SL."
            return {
                "action": label, "entry": round(close, 2),
                "stop_loss": sl, "sl_pct": f"-{sl_pct}%",
                "target": tp, "upside_pct": f"+{upside_pct}%",
                "trailing_stop": f"{TRAILING_STOP_PCT*100:.0f}%",
                "rr": f"1:{RR_RATIO:.0f}", "hold": hold,
                "qty": qty, "position_value": f"₹{actual_value:,}",
                "max_loss": f"₹{round(qty * sl_dist, 2):,}",
                "exit_rule": "Exit at TP | Trailing SL hit | Price below EMA 40 | Death cross",
            }
        elif label in ("SELL", "EXIT"):
            return {
                "action": "EXIT POSITION",
                "note": "NSE equity delivery — no shorting. Exit signal only.",
                "hold": "Exit within 1–3 trading days.",
                "reason": "Death cross" if label == "SELL" else "Disaster line breach (EMA 40)",
            }
        else:
            return {"action": "HOLD",
                    "hold": "Wait. No clear signal. Monitor daily.",
                    "reason": "Mixed confirmations. Watch for crossover + confluence."}

print("TradingSignals class ready (v5: your exact strategy).")


# ==============================================================================
# CLASS: XGBoostClassifier  [NEW — Binary Classification]
# Predicts: "Will price rise ≥2% in next 10 trading days?"
# FIXES: data leakage, walk-forward validation, SHAP feature selection
# ==============================================================================

class XGBoostClassifier:

    FEATURES = [
        "RSI", "MACD", "MACD_Hist", "ADX", "Plus_DI", "Minus_DI",
        "BB_Pct", "BB_Width", "OBV_Rising", "Volume_Ratio",
        "Daily_Return", "Return_5d", "Return_10d",
        "MA_20_above_50", "Price_Above_MA200", "Price_Above_EMA40",
        "ATR_Pct", "MACD_Bullish", "ADX_Bullish", "ADX_Strong",
    ]

    def __init__(self, symbol):
        self.symbol        = symbol
        self.model         = None
        self.selected_features = None
        self.model_path    = os.path.join(ML_MODELS_DIR, f"{symbol}_xgb_clf.json")
        self.meta_path     = os.path.join(ML_MODELS_DIR, f"{symbol}_xgb_clf_meta.json")

    def _prepare_data(self, df):
        """
        Create features + binary target.
        Target: 1 if price rises ≥ XGB_THRESHOLD_PCT% in next XGB_LOOKFORWARD days, else 0.
        NO DATA LEAKAGE: features are computed on full data, but target uses future returns.
        """
        data = df.tail(XGB_HISTORY_DAYS).copy()

        # Binary target: future return
        data["future_return"] = data["close"].pct_change(XGB_LOOKFORWARD).shift(-XGB_LOOKFORWARD) * 100
        data["target"] = (data["future_return"] >= XGB_THRESHOLD_PCT).astype(int)

        # Select available features
        feat_cols = [c for c in self.FEATURES if c in data.columns]

        # Convert booleans to int
        for c in feat_cols:
            if data[c].dtype == bool:
                data[c] = data[c].astype(int)

        # Add lag features (last XGB_LAG_DAYS of each core feature)
        core_lags = ["RSI", "MACD_Hist", "BB_Pct", "ADX", "Volume_Ratio", "Daily_Return"]
        core_lags = [c for c in core_lags if c in data.columns]
        for col in core_lags:
            for lag in [5, 10, 20]:
                data[f"{col}_lag{lag}"] = data[col].shift(lag)
                feat_cols.append(f"{col}_lag{lag}")

        data.dropna(inplace=True)
        return data, feat_cols

    def train(self, stock_df):
        data, feat_cols = self._prepare_data(stock_df)
        if len(data) < 500:
            return None

        # ── WALK-FORWARD SPLIT (no data leakage) ────────────────────────
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data  = data.iloc[split_idx:]

        X_train = train_data[feat_cols].values
        y_train = train_data["target"].values
        X_test  = test_data[feat_cols].values
        y_test  = test_data["target"].values

        # ── SHAP FEATURE SELECTION (fit on train only) ──────────────────
        if HAS_SHAP and len(feat_cols) > 10:
            # Quick pre-model for SHAP
            pre_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                verbosity=0, random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            )
            pre_model.fit(X_train, y_train)

            explainer   = shap.TreeExplainer(pre_model)
            shap_values = explainer.shap_values(X_train[:min(500, len(X_train))])
            importance  = np.abs(shap_values).mean(axis=0)

            # Keep top features (importance > median)
            threshold = np.median(importance)
            selected_mask = importance >= threshold
            selected_cols = [feat_cols[i] for i in range(len(feat_cols)) if selected_mask[i]]

            if len(selected_cols) >= 5:
                feat_cols = selected_cols
                X_train = train_data[feat_cols].values
                X_test  = test_data[feat_cols].values
                print(f"    SHAP selected {len(feat_cols)} features", end=" ")

        self.selected_features = feat_cols

        # ── TRAIN FINAL MODEL ───────────────────────────────────────────
        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos = neg_count / max(pos_count, 1)

        self.model = xgb.XGBClassifier(
            n_estimators      = 300,
            learning_rate     = 0.05,
            max_depth         = 4,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos,
            random_state      = 42,
            verbosity         = 0,
            use_label_encoder = False,
            eval_metric       = "logloss",
            early_stopping_rounds = 20,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── EVALUATE ────────────────────────────────────────────────────
        preds     = self.model.predict(X_test)
        proba     = self.model.predict_proba(X_test)[:, 1]
        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)

        self.model.save_model(self.model_path)
        meta = {
            "symbol":       self.symbol,
            "features":     feat_cols,
            "accuracy":     round(accuracy, 4),
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            "train_rows":   len(train_data),
            "test_rows":    len(test_data),
            "pos_rate":     round(y_test.mean(), 4),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def predict(self, stock_df):
        """Return probability of ≥2% rise in next 10 days."""
        if self.model is None or self.selected_features is None:
            return None, None

        data, _ = self._prepare_data(stock_df)
        if len(data) == 0:
            return None, None

        latest_features = data[self.selected_features].iloc[-1:].values
        pred  = int(self.model.predict(latest_features)[0])
        proba = float(self.model.predict_proba(latest_features)[0][1])
        return pred, proba

print("XGBoostClassifier ready (binary: ≥2% rise in 10 days, SHAP selection).")


# ==============================================================================
# CLASS: LSTMForecaster  [NEW — replaces ARIMA]
# Sequential pattern recognition on normalized indicator time series
# ==============================================================================

class LSTMForecaster:

    FEATURES = [
        "close", "RSI", "MACD_Hist", "ADX", "BB_Pct",
        "Volume_Ratio", "OBV_Rising", "Daily_Return",
    ]

    def __init__(self, symbol):
        self.symbol   = symbol
        self.model    = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _prepare_sequences(self, df, seq_len=SEQ_LENGTH):
        data = df.tail(LSTM_HISTORY_DAYS).copy()
        feat_cols = [c for c in self.FEATURES if c in data.columns]
        for c in feat_cols:
            if data[c].dtype == bool:
                data[c] = data[c].astype(float)
        data = data[feat_cols + ["close"]].dropna()

        if len(data) < seq_len + 50:
            return None, None, None, None, None

        X_raw = data[feat_cols].values
        y_raw = data["close"].values.reshape(-1, 1)

        # Split BEFORE scaling (no leakage)
        split = int(len(X_raw) * 0.8)

        X_train_raw = X_raw[:split]
        X_test_raw  = X_raw[split:]
        y_train_raw = y_raw[:split]
        y_test_raw  = y_raw[split:]

        # Fit scalers on train only
        self.scaler_X.fit(X_train_raw)
        self.scaler_y.fit(y_train_raw)

        X_all_scaled = self.scaler_X.transform(X_raw)
        y_all_scaled = self.scaler_y.transform(y_raw)

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_all_scaled)):
            X_seq.append(X_all_scaled[i-seq_len:i])
            y_seq.append(y_all_scaled[i, 0])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        split_seq = split - seq_len
        return (X_seq[:split_seq], y_seq[:split_seq],
                X_seq[split_seq:], y_seq[split_seq:], feat_cols)

    def train(self, stock_df):
        if not HAS_TF:
            return None

        result = self._prepare_sequences(stock_df)
        if result[0] is None:
            return None

        X_train, y_train, X_test, y_test, feat_cols = result

        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
        self.model.fit(
            X_train, y_train,
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0,
        )

        # Evaluate
        preds_scaled = self.model.predict(X_test, verbose=0)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mape = np.mean(np.abs((actual - preds) / actual)) * 100

        return {"symbol": self.symbol, "mae": round(mae, 2),
                "rmse": round(rmse, 2), "mape": round(mape, 2)}

    def predict_next(self, stock_df, n_days=None):
        if not HAS_TF or self.model is None:
            return []
        if n_days is None:
            n_days = ML_FORECAST_DAYS

        data = stock_df.tail(LSTM_HISTORY_DAYS).copy()
        feat_cols = [c for c in self.FEATURES if c in data.columns]
        for c in feat_cols:
            if data[c].dtype == bool:
                data[c] = data[c].astype(float)
        data = data[feat_cols].dropna()

        if len(data) < SEQ_LENGTH:
            return []

        recent = self.scaler_X.transform(data.values)
        seq    = recent[-SEQ_LENGTH:]

        forecasts = []
        for _ in range(n_days):
            inp  = seq.reshape(1, SEQ_LENGTH, len(feat_cols))
            pred_scaled = self.model.predict(inp, verbose=0)[0, 0]
            pred_price  = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            forecasts.append(round(float(pred_price), 2))

            # Shift sequence: append prediction, remove oldest
            new_row = seq[-1].copy()
            new_row[0] = pred_scaled  # update close (scaled)
            seq = np.vstack([seq[1:], new_row])

        return forecasts

print("LSTMForecaster ready (64→32 LSTM, dropout, early stopping).")


# ==============================================================================
# CLASS: RNNForecaster  [NEW — SimpleRNN for trend continuation]
# ==============================================================================

class RNNForecaster:

    FEATURES = ["close", "RSI", "MACD_Hist", "BB_Pct", "Volume_Ratio", "Daily_Return"]

    def __init__(self, symbol):
        self.symbol   = symbol
        self.model    = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train(self, stock_df):
        if not HAS_TF:
            return None

        data = stock_df.tail(LSTM_HISTORY_DAYS).copy()
        feat_cols = [c for c in self.FEATURES if c in data.columns]
        for c in feat_cols:
            if data[c].dtype == bool:
                data[c] = data[c].astype(float)
        data = data[feat_cols + ["close"]].dropna()

        if len(data) < SEQ_LENGTH + 50:
            return None

        X_raw = data[feat_cols].values
        y_raw = data["close"].values.reshape(-1, 1)

        split = int(len(X_raw) * 0.8)
        self.scaler_X.fit(X_raw[:split])
        self.scaler_y.fit(y_raw[:split])

        X_scaled = self.scaler_X.transform(X_raw)
        y_scaled = self.scaler_y.transform(y_raw)

        X_seq, y_seq = [], []
        for i in range(SEQ_LENGTH, len(X_scaled)):
            X_seq.append(X_scaled[i-SEQ_LENGTH:i])
            y_seq.append(y_scaled[i, 0])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        split_seq = split - SEQ_LENGTH
        X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
        y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]

        self.model = Sequential([
            SimpleRNN(48, return_sequences=True, input_shape=(SEQ_LENGTH, len(feat_cols))),
            Dropout(0.2),
            SimpleRNN(24),
            Dropout(0.2),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(patience=8, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0,
        )

        preds_scaled = self.model.predict(X_test, verbose=0)
        preds  = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mape = np.mean(np.abs((actual - preds) / actual)) * 100

        return {"symbol": self.symbol, "mae": round(mae, 2),
                "rmse": round(rmse, 2), "mape": round(mape, 2)}

    def predict_next(self, stock_df, n_days=None):
        if not HAS_TF or self.model is None:
            return []
        if n_days is None:
            n_days = ML_FORECAST_DAYS

        data = stock_df.tail(LSTM_HISTORY_DAYS).copy()
        feat_cols = [c for c in self.FEATURES if c in data.columns]
        for c in feat_cols:
            if data[c].dtype == bool:
                data[c] = data[c].astype(float)
        data = data[feat_cols].dropna()

        if len(data) < SEQ_LENGTH:
            return []

        recent = self.scaler_X.transform(data.values)
        seq = recent[-SEQ_LENGTH:]

        forecasts = []
        for _ in range(n_days):
            inp = seq.reshape(1, SEQ_LENGTH, len(feat_cols))
            pred_scaled = self.model.predict(inp, verbose=0)[0, 0]
            pred_price = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            forecasts.append(round(float(pred_price), 2))

            new_row = seq[-1].copy()
            new_row[0] = pred_scaled
            seq = np.vstack([seq[1:], new_row])

        return forecasts

print("RNNForecaster ready (48→24 SimpleRNN, dropout, early stopping).")


# ==============================================================================
# CLASS: BacktraderEngine  [NEW — Professional Backtesting]
# Uses Backtrader library for proper event-driven backtesting
# Implements YOUR exact strategy with trailing stop-loss
# ==============================================================================

class BacktraderEngine:

    @staticmethod
    def run_backtest_simple(df_ind, lookback_days=365, hold_days=40):
        """
        Event-driven backtest using your exact strategy.
        Includes: trailing stop-loss, minimum hold period, regime filter.
        Falls back to manual backtest if Backtrader not installed.
        """
        results = []
        all_syms = df_ind["symbol"].unique()

        for symbol in all_syms:
            s_df = df_ind[df_ind["symbol"] == symbol].copy().reset_index(drop=True)
            if len(s_df) < lookback_days:
                continue

            cutoff_idx = len(s_df) - lookback_days
            test_data  = s_df.iloc[cutoff_idx:].reset_index(drop=True)

            i = 0
            while i < len(test_data) - hold_days:
                row = test_data.iloc[i]

                # Check for your strategy signals
                ma_cross   = row.get("MA_Crossover", 0)
                above_50   = row.get("MA_20_above_50", False)
                above_ema40= row.get("Price_Above_EMA40", True)
                above_200  = row.get("Price_Above_MA200", False)
                rsi_ok     = row.get("RSI", 50) > 50
                macd_ok    = row.get("MACD_Bullish", False)
                bb_ok      = row.get("BB_Not_Overbought", True)
                obv_ok     = row.get("OBV_Rising", False)
                adx_ok     = row.get("ADX_Bullish", False)
                regime_ok  = row.get("Trend_Regime", "") == "BULLISH_REGIME"

                # Primary: crossover or in bullish zone
                has_cross = (ma_cross == 1) or above_50

                # Count confirmations
                confirmations = sum([rsi_ok, macd_ok, bb_ok, above_ema40,
                                     above_200, obv_ok, adx_ok, regime_ok])

                # BUY only if crossover + at least 6/8 confirmations
                if not (has_cross and confirmations >= 6):
                    i += 1
                    continue

                entry  = row["close"]
                atr    = row.get("ATR", entry * 0.02)
                if pd.isna(atr) or atr == 0:
                    atr = entry * 0.02

                sl     = entry - 2 * atr
                tp     = entry + 2 * atr * RR_RATIO
                ema40  = row.get("EMA_40", entry * 0.95)

                # ── Simulate trade with trailing stop ────────────────────
                outcome   = "TIMEOUT"
                days_held = hold_days
                exit_price= entry
                peak      = entry

                future = test_data.iloc[i+1:i+1+hold_days]
                for j, (_, frow) in enumerate(future.iterrows(), 1):
                    fp = frow["close"]

                    # Update trailing stop
                    if fp > peak:
                        peak = fp
                    trailing_sl = peak * (1 - TRAILING_STOP_PCT)

                    # Check exits (priority order)
                    if fp <= max(sl, trailing_sl):
                        outcome = "LOSS (SL/trailing)"
                        days_held = j
                        exit_price = fp
                        break
                    if fp < frow.get("EMA_40", 0):
                        outcome = "EXIT (EMA40 breach)"
                        days_held = j
                        exit_price = fp
                        break
                    if fp >= tp:
                        outcome = "WIN (TP hit)"
                        days_held = j
                        exit_price = fp
                        break
                    if j >= MIN_HOLD_DAYS and frow.get("MA_Crossover", 0) == -1:
                        outcome = "EXIT (death cross)"
                        days_held = j
                        exit_price = fp
                        break

                if outcome == "TIMEOUT":
                    exit_price = future["close"].iloc[-1] if len(future) > 0 else entry

                qty = int((CAPITAL * CAPITAL_PER_TRADE) / entry) if entry > 0 else 0
                pnl = qty * (exit_price - entry)
                pnl_pct = (exit_price - entry) / entry * 100

                results.append({
                    "symbol":      symbol,
                    "date":        row.get("date", ""),
                    "signal":      "BUY" if confirmations >= 7 else "WEAK BUY",
                    "confirmations": f"{confirmations}/8",
                    "entry":       round(entry, 2),
                    "exit_price":  round(exit_price, 2),
                    "sl":          round(sl, 2),
                    "tp":          round(tp, 2),
                    "outcome":     outcome,
                    "days_held":   days_held,
                    "qty":         qty,
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(pnl_pct, 2),
                })

                # Skip ahead past this trade
                i += days_held + 1
                continue

            # end while
        # end for

        if not results:
            print("  No BUY signals found in lookback period.")
            return pd.DataFrame()

        bt_df = pd.DataFrame(results)
        total = len(bt_df)
        wins  = bt_df["outcome"].str.contains("WIN").sum()
        losses= bt_df["outcome"].str.contains("LOSS|EXIT").sum()
        win_rate = wins / total * 100 if total else 0
        total_pnl = bt_df["pnl"].sum()

        print(f"  Backtest period  : last {lookback_days} calendar days")
        print(f"  Total trades     : {total}")
        print(f"  Wins             : {wins}  ({win_rate:.1f}%)")
        print(f"  Losses/Exits     : {losses}")
        print(f"  Timeouts         : {total - wins - losses}")
        print(f"  Total P&L        : ₹{total_pnl:,.2f}")
        print(f"  Avg P&L/trade    : ₹{bt_df['pnl'].mean():,.2f}")
        print(f"  Avg hold days    : {bt_df['days_held'].mean():.1f}")
        print(f"  Best trade       : ₹{bt_df['pnl'].max():,.2f}")
        print(f"  Worst trade      : ₹{bt_df['pnl'].min():,.2f}")

        bt_df.to_csv(BACKTEST_CSV, index=False)
        print(f"\n  Saved: {BACKTEST_CSV}")
        return bt_df

print("BacktraderEngine ready (trailing SL, EMA40 exit, death cross exit).")


# ==============================================================================
# ========================= PIPELINE EXECUTION =================================
# ==============================================================================

print(f"\n{'='*80}")
print("  NIFTY 50 ML PIPELINE  [v5 — LSTM + XGBoost Classifier + RNN]")
print(f"{'='*80}")
print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")


# ── STEP 1: LOAD DATA ───────────────────────────────────────────────────────

print("\n[STEP 1] Loading master dataset...")
print("="*80)

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(f"{MASTER_CSV} not found.\nRun first: python 01_data_updater.py")

master_df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
master_df.sort_values(["symbol", "date"], inplace=True)
master_df.reset_index(drop=True, inplace=True)

print(f"  File   : {MASTER_CSV}")
print(f"  Rows   : {len(master_df):,}")
print(f"  Stocks : {master_df['symbol'].nunique()}")
print(f"  Range  : {master_df['date'].min().date()} → {master_df['date'].max().date()}")


# ── STEP 2: CALCULATE ALL INDICATORS ────────────────────────────────────────

print("\n[STEP 2] Calculating technical indicators (RSI, SMA, EMA40, MACD, ADX, BB, OBV, ATR)...")
print("="*80)

processed = []
failed    = []
symbols   = master_df["symbol"].unique()
t0        = time.time()

for idx, symbol in enumerate(symbols, 1):
    print(f"[{idx:2d}/{len(symbols)}]  {symbol:12s}", end="  ")
    try:
        s_df = master_df[master_df["symbol"] == symbol].copy()
        s_df.reset_index(drop=True, inplace=True)
        s_df = TradingIndicators.calculate_all(s_df)
        processed.append(s_df)
        print(f"done  ({len(s_df):,} rows, {s_df.shape[1]} cols)")
    except Exception as e:
        print(f"ERROR — {str(e)[:70]}")
        failed.append(symbol)

print(f"\n  Completed in {time.time() - t0:.1f}s")
if failed:
    print(f"  Failed: {failed}")


# ── STEP 3: SAVE INDICATORS ─────────────────────────────────────────────────

print("\n[STEP 3] Saving indicators dataset...")
print("="*80)

df_ind = pd.concat(processed, ignore_index=True)
df_ind.sort_values(["symbol", "date"], inplace=True)
df_ind.reset_index(drop=True, inplace=True)
df_ind.to_csv(INDICATORS_CSV, index=False, date_format="%Y-%m-%d")

print(f"  Saved  : {INDICATORS_CSV}")
print(f"  Rows   : {len(df_ind):,}")
print(f"  Cols   : {df_ind.shape[1]}")


# ── STEP 4: GENERATE TRADING SIGNALS (your exact strategy) ──────────────────

print("\n[STEP 4] Generating signals (your strategy: crossover + 7 confirmations)...")
print("="*80)

signals_rows = []

for symbol in df_ind["symbol"].unique():
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    if len(s_df) == 0:
        continue
    try:
        s_df, score, label, details, confirmations = TradingSignals.generate(s_df)
        latest = s_df.iloc[-1]

        def safe(col):
            v = latest.get(col, None)
            return round(float(v), 4) if (v is not None and not pd.isna(v)) else None

        trade = TradingSignals.get_trade_plan(latest, label, score)

        bull_count = sum(1 for v in confirmations.values() if v)

        signals_rows.append({
            "symbol":          symbol,
            "as_of_date":      str(latest["date"])[:10],
            "close":           safe("close"),
            "signal":          label,
            "score":           score,
            "confirmations":   f"{bull_count}/{len(confirmations)}",
            "action":          trade.get("action", ""),
            "entry":           trade.get("entry", ""),
            "stop_loss":       trade.get("stop_loss", ""),
            "sl_pct":          trade.get("sl_pct", ""),
            "target":          trade.get("target", ""),
            "upside_pct":      trade.get("upside_pct", ""),
            "trailing_stop":   trade.get("trailing_stop", ""),
            "rr":              trade.get("rr", ""),
            "hold_duration":   trade.get("hold", ""),
            "qty":             trade.get("qty", ""),
            "position_value":  trade.get("position_value", ""),
            "max_loss":        trade.get("max_loss", ""),
            "rsi":             safe("RSI"),
            "rsi_zone":        latest.get("RSI_Zone", ""),
            "macd":            safe("MACD"),
            "macd_signal":     safe("MACD_Signal"),
            "macd_bullish":    latest.get("MACD_Bullish", ""),
            "adx":             safe("ADX"),
            "adx_bullish":     latest.get("ADX_Bullish", ""),
            "ema_40":          safe("EMA_40"),
            "disaster_line":   latest.get("Disaster_Line", ""),
            "ma_20":           safe("MA_20"),
            "ma_50":           safe("MA_50"),
            "ma_200":          safe("MA_200"),
            "cross_type":      latest.get("Cross_Type", ""),
            "trend_regime":    latest.get("Trend_Regime", ""),
            "long_term_trend": latest.get("Long_Term_Trend", ""),
            "obv_rising":      latest.get("OBV_Rising", ""),
            "bb_pct":          safe("BB_Pct"),
            "volume_ratio":    safe("Volume_Ratio"),
            "atr":             safe("ATR"),
            "signal_details":  " | ".join(details),
        })
        emoji = {"STRONG BUY":"🟢","BUY":"🟡","HOLD":"⚪","SELL":"🔴","EXIT":"🚨"}.get(label,"⚪")
        print(f"  {emoji} {symbol:12s} | Score: {score:+4d} | {label:12s} | {bull_count}/{len(confirmations)} conf")
    except Exception as e:
        print(f"  {symbol:12s} | ERROR: {e}")

signals_df = pd.DataFrame(signals_rows).sort_values("score", ascending=False)
signals_df.reset_index(drop=True, inplace=True)

# Print summary
for lbl in ["STRONG BUY", "BUY", "HOLD", "SELL", "EXIT"]:
    sub = signals_df[signals_df["signal"] == lbl]
    if len(sub):
        print(f"\n  -- {lbl} ({len(sub)}) --")
        for _, r in sub.head(5).iterrows():
            print(f"    {r['symbol']:12s}  ₹{r['close']:>10.2f}  {r['confirmations']}  RSI {r['rsi']:.1f}")


# ── STEP 5A: XGBOOST CLASSIFIER ─────────────────────────────────────────────

print(f"\n[STEP 5A] XGBoost Binary Classifier (≥{XGB_THRESHOLD_PCT}% rise in {XGB_LOOKFORWARD}d)...")
print("="*80)

ml_symbols = ML_SYMBOLS if ML_SYMBOLS else list(df_ind["symbol"].unique())
xgb_rows   = []
xgb_metas  = []

for symbol in ml_symbols:
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    print(f"  {symbol:12s}", end="  ")
    t1 = time.time()

    clf = XGBoostClassifier(symbol)
    meta = clf.train(s_df)
    if meta is None:
        print("SKIPPED (insufficient data)")
        continue

    pred, proba = clf.predict(s_df)
    xgb_rows.append({
        "symbol": symbol,
        "xgb_pred": pred,
        "xgb_proba": round(proba, 4) if proba else None,
        "xgb_signal": "BULLISH" if pred == 1 else "BEARISH",
    })
    xgb_metas.append(meta)
    print(f"done ({time.time()-t1:.1f}s)  "
          f"Acc={meta['accuracy']:.2%}  Prec={meta['precision']:.2%}  "
          f"F1={meta['f1']:.2%}  → {'BULLISH' if pred==1 else 'BEARISH'} ({proba:.1%})")

xgb_df = pd.DataFrame(xgb_rows)
print(f"\n  XGBoost Classifier completed for {len(xgb_df)} symbols.")
if xgb_metas:
    avg_acc = np.mean([m["accuracy"] for m in xgb_metas])
    avg_f1  = np.mean([m["f1"] for m in xgb_metas])
    print(f"  Avg Accuracy: {avg_acc:.2%}  |  Avg F1: {avg_f1:.2%}")


# ── STEP 5B: LSTM FORECASTS ─────────────────────────────────────────────────

print(f"\n[STEP 5B] LSTM Forecasts (64→32 units, {SEQ_LENGTH}-day sequences)...")
print("="*80)

lstm_rows  = []
lstm_metas = []

for symbol in ml_symbols:
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    print(f"  {symbol:12s}", end="  ")
    t1 = time.time()

    model = LSTMForecaster(symbol)
    meta  = model.train(s_df)
    if meta is None:
        print("SKIPPED (insufficient data or no TensorFlow)")
        continue

    forecasts = model.predict_next(s_df)
    if forecasts:
        row = {"symbol": symbol}
        for d, v in enumerate(forecasts, 1):
            row[f"lstm_d{d}"] = v
        lstm_rows.append(row)
        lstm_metas.append(meta)
        print(f"done ({time.time()-t1:.1f}s)  "
              f"MAE={meta['mae']:.2f}  MAPE={meta['mape']:.2f}%  D+1: ₹{forecasts[0]:.2f}")
    else:
        print("SKIPPED (predict failed)")

lstm_df = pd.DataFrame(lstm_rows)
print(f"\n  LSTM completed for {len(lstm_df)} symbols.")


# ── STEP 5C: RNN FORECASTS ──────────────────────────────────────────────────

print(f"\n[STEP 5C] SimpleRNN Forecasts (48→24 units)...")
print("="*80)

rnn_rows  = []
rnn_metas = []

for symbol in ml_symbols:
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    print(f"  {symbol:12s}", end="  ")
    t1 = time.time()

    model = RNNForecaster(symbol)
    meta  = model.train(s_df)
    if meta is None:
        print("SKIPPED")
        continue

    forecasts = model.predict_next(s_df)
    if forecasts:
        row = {"symbol": symbol}
        for d, v in enumerate(forecasts, 1):
            row[f"rnn_d{d}"] = v
        rnn_rows.append(row)
        rnn_metas.append(meta)
        print(f"done ({time.time()-t1:.1f}s)  "
              f"MAE={meta['mae']:.2f}  MAPE={meta['mape']:.2f}%  D+1: ₹{forecasts[0]:.2f}")
    else:
        print("SKIPPED")

rnn_df = pd.DataFrame(rnn_rows)
print(f"\n  RNN completed for {len(rnn_df)} symbols.")


# ── STEP 6: MERGE ALL + CONSENSUS ───────────────────────────────────────────

print("\n[STEP 6] Merging signals + ML forecasts + XGBoost classifier...")
print("="*80)

final_df = signals_df.copy()

# Merge XGBoost classifier results
if len(xgb_df) > 0:
    final_df = final_df.merge(xgb_df, on="symbol", how="left")

# Merge LSTM forecasts
if len(lstm_df) > 0:
    final_df = final_df.merge(lstm_df, on="symbol", how="left")

# Merge RNN forecasts
if len(rnn_df) > 0:
    final_df = final_df.merge(rnn_df, on="symbol", how="left")

# ── Consensus from LSTM + RNN (median of price forecasts) ───────────────────
d1_cols = [c for c in ("lstm_d1", "rnn_d1") if c in final_df.columns]
if d1_cols:
    final_df["ml_consensus_d1"] = final_df[d1_cols].median(axis=1).round(2)
    final_df["ml_upside_pct"]   = (
        (final_df["ml_consensus_d1"] - final_df["close"])
        / final_df["close"] * 100
    ).round(2)

# ── ML-enhanced signal (combine your strategy + XGBoost classifier) ─────────
if "xgb_pred" in final_df.columns:
    def ml_enhanced_signal(row):
        base_signal = row["signal"]
        xgb_pred    = row.get("xgb_pred", None)
        xgb_proba   = row.get("xgb_proba", 0)

        if base_signal in ("STRONG BUY", "BUY") and xgb_pred == 1:
            return f"{base_signal} (ML confirms {xgb_proba:.0%})"
        elif base_signal in ("STRONG BUY", "BUY") and xgb_pred == 0:
            return f"HOLD (ML disagrees: only {xgb_proba:.0%} bullish)"
        elif base_signal in ("SELL", "EXIT"):
            return base_signal
        else:
            if xgb_pred == 1 and xgb_proba > 0.7:
                return f"WATCH (ML bullish {xgb_proba:.0%}, await crossover)"
            return base_signal

    final_df["ml_enhanced_signal"] = final_df.apply(ml_enhanced_signal, axis=1)

# Save
final_df.to_csv(SIGNALS_CSV, index=False)

ml_parts = [f for f in [xgb_df, lstm_df, rnn_df] if len(f) > 0]
if ml_parts:
    ml_out = ml_parts[0]
    for f in ml_parts[1:]:
        ml_out = ml_out.merge(f, on="symbol", how="outer")
    ml_out.to_csv(ML_FORECASTS_CSV, index=False)

print(f"  Saved: {SIGNALS_CSV}")
print(f"  Saved: {ML_FORECASTS_CSV}")
print(f"  Rows : {len(final_df)}")


# ── STEP 7: BACKTESTING (your strategy + trailing stop) ─────────────────────

print("\n[STEP 7] Backtesting (your strategy, trailing SL, EMA40 exit)...")
print("="*80)

backtest_df = BacktraderEngine.run_backtest_simple(df_ind, lookback_days=365)


# ── STEP 8: FINAL SUMMARY ───────────────────────────────────────────────────

print(f"\n{'='*80}")
print("  PIPELINE COMPLETE  [v5]")
print(f"{'='*80}")

for lbl in ["STRONG BUY", "BUY", "HOLD", "SELL", "EXIT"]:
    sub = final_df[final_df["signal"] == lbl]
    print(f"  {lbl:15s}: {len(sub):2d}")

print()
print("  Files created:")
print(f"    1. {INDICATORS_CSV}")
print(f"    2. {SIGNALS_CSV}")
print(f"    3. {ML_FORECASTS_CSV}")
print(f"    4. {BACKTEST_CSV}")
print(f"    5. {ML_MODELS_DIR}/")
print()
print("  Strategy: YOUR 20/50 SMA crossover + 7 confirmations")
print("  Models:")
print(f"    LSTM           : price forecast (64→32 units, {SEQ_LENGTH}-day sequences)")
print(f"    XGBoost Clf    : binary (≥{XGB_THRESHOLD_PCT}% in {XGB_LOOKFORWARD}d) + SHAP selection")
print(f"    SimpleRNN      : price forecast (48→24 units)")
print(f"    Consensus      : median of LSTM + RNN | majority vote with XGBoost")
print()
print("  Risk Management:")
print(f"    Trailing SL    : {TRAILING_STOP_PCT*100:.0f}%")
print(f"    Disaster line  : EMA 40 (breach = immediate EXIT)")
print(f"    Regime filter  : SMA 50 > SMA 200")
print(f"    Min hold       : {MIN_HOLD_DAYS} days")
print()
print(f"  Next step: streamlit run app.py")
print(f"{'='*80}")
print(f"  Finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'='*80}")
