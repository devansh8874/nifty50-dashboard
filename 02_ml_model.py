# ==============================================================================
#  NIFTY 50 — ML FORECASTING PIPELINE  [v4]
#  MBA Dissertation: AI-Driven Algorithmic Trading System
# ------------------------------------------------------------------------------
#  CHANGES FROM PREVIOUS VERSION:
#    - REMOVED: LSTM (broken MAPE=inf, too slow, scaler mismatch)
#    - ADDED:   XGBoost forecaster (learns bullish indicator patterns)
#    - MODELS:  ARIMA + Prophet + XGBoost  (3 models — odd number for median)
#    - CONSENSUS: median of 3 forecasts (robust to one outlier model)
#    - Features: close + RSI + MA20/50/200 + OBV + BB + Volume = 13 features
#    - XGBoost learns YOUR manual trading logic from indicator combinations
# ==============================================================================


import pandas as pd
import numpy as np
import os, warnings, json, time
from datetime import datetime, timedelta

# ── Install dependencies ──────────────────────────────────────────────────────
# Run once: pip install pmdarima prophet xgboost scikit-learn
# pip install pmdarima
# pip install prophet
# pip install xgboost

import pmdarima as pm
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

print(f"pandas    : {pd.__version__}")
print(f"pmdarima  : {pm.__version__}")
print(f"xgboost   : {xgb.__version__}")
print("All imports OK.")


# ==============================================================================
# CONFIGURATION  ← only change values here, nothing else needs editing
# ==============================================================================

MASTER_CSV          = "02_data/nifty50_historical_master.csv"
INDICATORS_CSV      = "02_data/nifty50_with_indicators.csv"
SIGNALS_CSV         = "02_data/latest_signals_summary.csv"
ML_FORECASTS_CSV    = "02_data/ml_forecasts.csv"
BACKTEST_CSV        = "02_data/backtest_results.csv"
ML_MODELS_DIR       = "02_data/ml_models"

os.makedirs("02_data", exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# Set ML_SYMBOLS to a list to run on select stocks only
# e.g. ML_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK"]
ML_SYMBOLS          = None           # None = all 50 stocks

ML_FORECAST_DAYS    = 84         # predict next 5 trading days
ARIMA_HISTORY_DAYS  = 1008        # ~3 years
PROPHET_HISTORY_DAYS= 2520           # ~5 years
XGB_LAG_DAYS        = 60            # XGBoost uses last 20 days as features
XGB_HISTORY_DAYS    = 2520          # ~5 years for XGBoost training
RETRAIN_XGB         = True  # False = use cached .json model if exists

# Your manual trading rules (used in signal generation + backtest)
CAPITAL             = 100_000        # INR
CAPITAL_PER_TRADE   = 0.20           # 20% per trade
MAX_TRADES          = 2              # never more than 2 open trades
RR_RATIO            = 2.0            # 1:2 risk-reward

print("Configuration loaded.")
print(f"  ML_SYMBOLS     : {ML_SYMBOLS if ML_SYMBOLS else 'ALL 50'}")
print(f"  Forecast days  : {ML_FORECAST_DAYS}")
print(f"  Capital/trade  : ₹{CAPITAL * CAPITAL_PER_TRADE:,.0f}")


# ==============================================================================
# CLASS: TradingIndicators
# Your 5 manual indicators: RSI, SMA20/50, SMA200, OBV, Bollinger Bands
# ==============================================================================

class TradingIndicators:

    @staticmethod
    def add_rsi(df, period=14):
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"]          = 100 - (100 / (1 + rs))
        df["RSI_Above_60"] = df["RSI"] > 60
        df["RSI_Below_40"] = df["RSI"] < 40
        df["RSI_Zone"]     = np.where(df["RSI"] > 60, "OVERBOUGHT",
                             np.where(df["RSI"] < 40, "OVERSOLD", "NEUTRAL"))
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
        df["Return_20d"]   = df["close"].pct_change(20) * 100
        return df

    @classmethod
    def calculate_all(cls, df):
        df = cls.add_rsi(df)
        df = cls.add_sma_cross(df)
        df = cls.add_sma200(df)
        df = cls.add_bollinger(df)
        df = cls.add_obv(df)
        df = cls.add_volume(df)
        df = cls.add_returns(df)
        return df

print("TradingIndicators class ready.")


# ==============================================================================
# CLASS: TradingSignals
# Mirrors your exact manual trading logic:
#   RSI 60/40 | SMA 20/50 crossover | SMA 200 trend | OBV | Bollinger Bands
#   No shorting | Equity delivery swing | RR 1:2
# ==============================================================================

class TradingSignals:

    @classmethod
    def generate(cls, stock_df):
        if len(stock_df) == 0:
            return stock_df, 0, "NEUTRAL/HOLD", []

        latest  = stock_df.iloc[-1]
        score   = 0
        details = []

        # 1. RSI — weight 25
        rsi = latest.get("RSI", 50)
        if pd.isna(rsi): rsi = 50
        if rsi < 40:
            score += 25
            details.append(f"RSI {rsi:.1f} OVERSOLD <40 (+25)")
        elif rsi > 60:
            score -= 25
            details.append(f"RSI {rsi:.1f} OVERBOUGHT >60 (-25)")
        else:
            details.append(f"RSI {rsi:.1f} neutral 40-60 (0)")

        # 2. SMA 20/50 crossover — weight 30
        ma_cross = latest.get("MA_Crossover", 0)
        above_50 = latest.get("MA_20_above_50", False)
        if ma_cross == 1:
            score += 30; details.append("GOLDEN CROSS: 20SMA crossed above 50SMA (+30)")
        elif ma_cross == -1:
            score -= 30; details.append("DEATH CROSS: 20SMA crossed below 50SMA (-30)")
        elif above_50:
            score += 15; details.append("20SMA above 50SMA: bullish zone (+15)")
        else:
            score -= 15; details.append("20SMA below 50SMA: bearish zone (-15)")

        # 3. SMA 200 long-term trend — weight 20
        above_200 = latest.get("Price_Above_MA200", False)
        ma200     = latest.get("MA_200", 0)
        if above_200:
            score += 20; details.append(f"Price above 200SMA [{ma200:.1f}] BULL (+20)")
        else:
            score -= 20; details.append(f"Price below 200SMA [{ma200:.1f}] BEAR (-20)")

        # 4. OBV confirmation — weight 15
        obv_rising = latest.get("OBV_Rising", False)
        if obv_rising:
            score += 15; details.append("OBV above OBV_MA20: volume confirms uptrend (+15)")
        else:
            score -= 15; details.append("OBV below OBV_MA20: volume confirms downtrend (-15)")

        # 5. Bollinger Bands — weight 10
        close    = latest.get("close", 0)
        bb_upper = latest.get("BB_Upper", np.nan)
        bb_lower = latest.get("BB_Lower", np.nan)
        bb_mid   = latest.get("BB_Mid",   np.nan)
        if not any(pd.isna(x) for x in [bb_upper, bb_lower, bb_mid]):
            if close <= bb_lower:
                score += 10; details.append("Price at/below BB Lower: oversold bounce (+10)")
            elif close >= bb_upper:
                score -= 10; details.append("Price at/above BB Upper: overbought (-10)")
            elif bb_mid <= close <= bb_upper:
                score += 5;  details.append("Price in BB upper half: bullish momentum (+5)")
            else:
                score -= 5;  details.append("Price in BB lower half: weak momentum (-5)")

        # Strong combined signal bonus
        high_vol = latest.get("High_Volume", False)
        strong   = sum([(ma_cross == 1), (rsi < 55), obv_rising,
                        (not pd.isna(bb_lower) and close <= bb_lower * 1.02), high_vol])
        if strong >= 4:
            score += 10; details.append(f"STRONG COMBINED SIGNAL: {strong}/5 met (+10)")

        # Classify
        if   score >= 60:  label = "STRONG BUY"
        elif score >= 25:  label = "BUY"
        elif score <= -60: label = "STRONG SELL"
        elif score <= -25: label = "SELL"
        else:              label = "NEUTRAL/HOLD"

        stock_df = stock_df.copy()
        stock_df["Signal_Score"]   = score
        stock_df["Overall_Signal"] = label
        return stock_df, score, label, details

    @staticmethod
    def get_trade_plan(latest_row, label, score):
        """
        Trade plan based on your rules:
        - Capital: ₹1,00,000 | 20% per trade = ₹20,000 per position
        - Max 2 trades at once
        - SL = BB_Width / 2 below entry | TP = 2x SL (RR 1:2)
        - No shorting — equity delivery only
        """
        close    = latest_row.get("close", 0)
        bb_upper = latest_row.get("BB_Upper", np.nan)
        bb_lower = latest_row.get("BB_Lower", np.nan)
        above_200= latest_row.get("Price_Above_MA200", False)

        # SL/TP from Bollinger Band width (natural volatility measure)
        if not any(pd.isna(x) for x in [bb_upper, bb_lower]):
            bb_width = bb_upper - bb_lower
            sl_dist  = bb_width / 2
            tp_dist  = bb_width          # RR 1:2
        else:
            sl_dist  = close * 0.03
            tp_dist  = close * 0.06

        sl  = round(close - sl_dist, 2)
        tp  = round(close + tp_dist, 2)
        upside_pct = round((tp - close) / close * 100, 2)
        sl_pct     = round((close - sl) / close * 100, 2)

        # Position sizing: 20% of ₹1,00,000 = ₹20,000
        position_value = CAPITAL * CAPITAL_PER_TRADE
        qty            = int(position_value / close) if close > 0 else 0
        actual_value   = round(qty * close, 2)

        if label in ("BUY", "STRONG BUY"):
            if label == "STRONG BUY" and above_200:
                hold = "8–12 weeks. Exit at TP or 2 consecutive closes below BB_Lower."
            elif above_200:
                hold = "4–8 weeks. Re-check signals weekly."
            else:
                hold = "2–4 weeks. Cautious — price below 200SMA. Tight SL."
            return {
                "action": label, "entry": round(close, 2),
                "stop_loss": sl, "sl_pct": f"-{sl_pct}%",
                "target": tp,    "upside_pct": f"+{upside_pct}%",
                "rr": "1:2", "hold": hold,
                "qty": qty, "position_value": f"₹{actual_value:,}",
                "max_loss": f"₹{round(qty * sl_dist, 2):,}",
                "exit_rule": "Exit at TP | Exit at SL | RSI>70 AND OBV turns down",
            }
        elif label in ("SELL", "STRONG SELL"):
            return {
                "action": "EXIT POSITION",
                "note": "NSE equity delivery — no shorting. Exit signal only.",
                "hold": "Exit within 1–3 trading days.",
            }
        else:
            hold = ("Hold 10–15 more days." if score > 0 and above_200
                    else "Wait 5–7 days. No clear signal.")
            return {"action": "HOLD", "hold": hold,
                    "reason": "Mixed signals. Monitor RSI and OBV."}

print("TradingSignals class ready.")


# ==============================================================================
# CLASS: ARIMAForecaster
# - Univariate: close price only (ARIMA cannot take additional features)
# - Auto-selects best (p,d,q) using AIC criterion
# - 3 years of data for stability
# ==============================================================================

class ARIMAForecaster:

    @staticmethod
    def forecast(close_series, n_periods=None, history_days=None):
        if n_periods    is None: n_periods    = ML_FORECAST_DAYS
        if history_days is None: history_days = ARIMA_HISTORY_DAYS

        series = close_series.dropna().tail(history_days)
        if len(series) < 100:
            return None, None

        try:
            model = pm.auto_arima(
                series, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=5, max_q=5, information_criterion="aic",
            )
            fc, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            return fc, conf_int
        except Exception:
            return None, None

print("ARIMAForecaster class ready.")


# ==============================================================================
# CLASS: ProphetForecaster
# - Uses close as y (target)
# - Adds RSI and Volume_Ratio as extra regressors
#   → These two indicators are most correlated with your entry signals
# - 5 years of data for seasonality detection
# ==============================================================================

class ProphetForecaster:

    @staticmethod
    def forecast(stock_df, n_periods=None, history_days=None):
        if n_periods    is None: n_periods    = ML_FORECAST_DAYS
        if history_days is None: history_days = PROPHET_HISTORY_DAYS

        df = stock_df.dropna(subset=["close"]).tail(history_days).copy()
        if len(df) < 200:
            return None

        p_df    = pd.DataFrame({"ds": df["date"].values, "y": df["close"].values})
        has_vol = "Volume_Ratio" in df.columns
        has_rsi = "RSI" in df.columns and not df["RSI"].isna().all()

        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        if has_vol:
            p_df["volume_ratio"] = df["Volume_Ratio"].fillna(1.0).values
            m.add_regressor("volume_ratio")
        if has_rsi:
            p_df["rsi"] = df["RSI"].fillna(50.0).values
            m.add_regressor("rsi")

        try:
            m.fit(p_df, algorithm="LBFGS")
            future = m.make_future_dataframe(periods=n_periods, freq="B")
            if has_vol:
                future["volume_ratio"] = p_df["volume_ratio"].mean()
            if has_rsi:
                future["rsi"] = p_df["rsi"].iloc[-1]
            fcast = m.predict(future)
            return fcast.tail(n_periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        except Exception:
            return None

print("ProphetForecaster class ready.")


# ==============================================================================
# CLASS: XGBoostForecaster  ← REPLACES LSTM
# ------------------------------------------------------------------------------
# WHY XGBoost LEARNS BULLISH PATTERNS:
#   - Features include: RSI, MA20, MA50, MA200, OBV, BB_Pct, Volume_Ratio
#   - These are exactly YOUR manual indicators
#   - XGBoost creates lag features (last 20 days of each indicator)
#   - When RSI<40 + OBV rising + price near BB_Lower appear together in lags,
#     the model learns that price tends to rise → this IS bullishness detection
#   - 13 features × 20 lags = 260 input features per prediction
#
# WHY CLOSE + INDICATORS (not close alone):
#   - Close alone: model sees 1348.10 — no context
#   - Close + RSI + OBV + BB: model sees oversold + volume rising + near support
#   - The combination = the pattern your eyes look for on a chart
# ==============================================================================

class XGBoostForecaster:

    # These are your 5 manual indicators + supporting features
    FEATURES = [
        "close",                              # target series
        "RSI",                                # momentum
        "MA_20", "MA_50", "MA_200",           # trend
        "OBV_Rising", "Volume_Ratio",         # volume
        "BB_Pct", "BB_Width",                 # volatility
        "Daily_Return", "Return_5d",          # momentum returns
        "MA_20_above_50", "Price_Above_MA200" # crossover flags (0/1)
    ]

    def __init__(self, symbol):
        self.symbol       = symbol
        self.model        = None
        self.feature_cols = None
        self.model_path   = os.path.join(ML_MODELS_DIR, f"{symbol}_xgb.json")
        self.meta_path    = os.path.join(ML_MODELS_DIR, f"{symbol}_xgb_meta.json")
        self.n_lags       = XGB_LAG_DAYS

    def _make_lag_features(self, df):
        """
        For each indicator, create n_lags (20) lagged columns.
        This gives XGBoost the last 20 days of each indicator as context.
        Target y = next day close price.
        """
        cols = [c for c in self.FEATURES if c in df.columns]
        self.feature_cols = cols
        data = df[cols].dropna().tail(XGB_HISTORY_DAYS).copy()

        # Convert bool columns to int
        for c in ["OBV_Rising", "MA_20_above_50", "Price_Above_MA200", "High_Volume"]:
            if c in data.columns:
                data[c] = data[c].astype(int)

        lag_df = pd.DataFrame(index=data.index)
        for col in cols:
            for lag in range(1, self.n_lags + 1):
                lag_df[f"{col}_lag{lag}"] = data[col].shift(lag)

        lag_df["target"] = data["close"].shift(-1)   # next day close
        lag_df.dropna(inplace=True)
        return lag_df, data

    def train(self, stock_df, force=False):
        # Use cached model if available
        if os.path.exists(self.model_path) and not force and not RETRAIN_XGB:
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta["feature_cols"]
            return meta

        lag_df, _ = self._make_lag_features(stock_df)
        if len(lag_df) < 200:
            return None

        X = lag_df.drop("target", axis=1)
        y = lag_df["target"]

        split  = int(len(X) * 0.8)
        X_tr,  X_val  = X.iloc[:split],  X.iloc[split:]
        y_tr,  y_val  = y.iloc[:split],  y.iloc[split:]

        self.model = xgb.XGBRegressor(
            n_estimators      = 300,
            learning_rate     = 0.05,
            max_depth         = 4,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            random_state      = 42,
            verbosity         = 0,
            early_stopping_rounds = 20,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set    = [(X_val, y_val)],
            verbose     = False,
        )

        preds = self.model.predict(X_val)
        mae   = float(mean_absolute_error(y_val, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
        mape  = float(np.mean(np.abs((y_val.values - preds) / y_val.values)) * 100)

        self.model.save_model(self.model_path)
        meta = {
            "symbol":       self.symbol,
            "feature_cols": self.feature_cols,
            "mae":          round(mae, 2),
            "rmse":         round(rmse, 2),
            "mape":         round(mape, 2),
            "train_rows":   len(lag_df),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def predict_next(self, stock_df, n_days=None):
        """
        Iterative one-step-ahead forecast:
        Each predicted close is fed back as input for the next day.
        This is how you'd trade — you only have today's data to predict tomorrow.
        """
        if n_days is None: n_days = ML_FORECAST_DAYS
        if self.model is None or self.feature_cols is None:
            return []

        cols = [c for c in self.feature_cols if c in stock_df.columns]
        data = stock_df[cols].dropna().tail(self.n_lags + 5).copy()

        # Convert bool columns to int
        for c in ["OBV_Rising", "MA_20_above_50", "Price_Above_MA200"]:
            if c in data.columns:
                data[c] = data[c].astype(int)

        forecasts = []
        for _ in range(n_days):
            row = {}
            for col in cols:
                for lag in range(1, self.n_lags + 1):
                    idx = -(lag)
                    row[f"{col}_lag{lag}"] = data[col].iloc[idx] if len(data) >= abs(idx) else 0
            X    = pd.DataFrame([row])
            pred = float(self.model.predict(X)[0])
            forecasts.append(round(pred, 2))

            # Append predicted close for next iteration
            new_row           = data.iloc[-1].copy()
            new_row["close"]  = pred
            data = pd.concat([data, new_row.to_frame().T], ignore_index=True)

        return forecasts

print("XGBoostForecaster class ready.")
print()
print("NOTE: XGBoost replaces LSTM. Reasons:")
print("  1. LSTM gave MAPE=inf% — broken scaler-model mismatch")
print("  2. XGBoost is 50-100x faster (seconds vs minutes per stock)")
print("  3. XGBoost learns your indicator patterns (RSI+OBV+BB combinations)")
print("  4. Still 3 models (odd) → median consensus avoids ties")


# ==============================================================================
# STEP 1: LOAD MASTER CSV
# ==============================================================================

print(f"\n{'='*80}")
print("  NIFTY 50 FEATURE ENGINEERING  [v4 — ARIMA + Prophet + XGBoost]")
print(f"{'='*80}")
print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")

print("\n[STEP 1] Loading master dataset...")
print("="*80)

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(
        f"{MASTER_CSV} not found.\nRun first: python 01_data_updater.py"
    )

master_df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
master_df.sort_values(["symbol", "date"], inplace=True)
master_df.reset_index(drop=True, inplace=True)

print(f"  File   : {MASTER_CSV}")
print(f"  Rows   : {len(master_df):,}")
print(f"  Stocks : {master_df['symbol'].nunique()}")
print(f"  Range  : {master_df['date'].min().date()} → {master_df['date'].max().date()}")


# ==============================================================================
# STEP 2: CALCULATE TECHNICAL INDICATORS
# ==============================================================================

print("\n[STEP 2] Calculating technical indicators...")
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


# ==============================================================================
# STEP 3: SAVE INDICATORS CSV
# ==============================================================================

print("\n[STEP 3] Saving indicators dataset...")
print("="*80)

df_ind = pd.concat(processed, ignore_index=True)
df_ind.sort_values(["symbol", "date"], inplace=True)
df_ind.reset_index(drop=True, inplace=True)
df_ind.to_csv(INDICATORS_CSV, index=False, date_format="%Y-%m-%d")

print(f"  Saved  : {INDICATORS_CSV}")
print(f"  Rows   : {len(df_ind):,}")
print(f"  Cols   : {df_ind.shape[1]}")


# ==============================================================================
# STEP 4: GENERATE TRADING SIGNALS
# ==============================================================================

print("\n[STEP 4] Generating trading signals...")
print("="*80)

signals_rows = []

for symbol in df_ind["symbol"].unique():
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    if len(s_df) == 0:
        continue
    try:
        s_df, score, label, details = TradingSignals.generate(s_df)
        latest = s_df.iloc[-1]

        def safe(col):
            v = latest.get(col, None)
            return round(float(v), 4) if (v is not None and not pd.isna(v)) else None

        trade = TradingSignals.get_trade_plan(latest, label, score)

        signals_rows.append({
            "symbol":          symbol,
            "as_of_date":      str(latest["date"])[:10],
            "close":           safe("close"),
            "signal":          label,
            "score":           score,
            "action":          trade.get("action", ""),
            "entry":           trade.get("entry", ""),
            "stop_loss":       trade.get("stop_loss", ""),
            "sl_pct":          trade.get("sl_pct", ""),
            "target":          trade.get("target", ""),
            "upside_pct":      trade.get("upside_pct", ""),
            "rr":              trade.get("rr", ""),
            "hold_duration":   trade.get("hold", ""),
            "qty":             trade.get("qty", ""),
            "position_value":  trade.get("position_value", ""),
            "max_loss":        trade.get("max_loss", ""),
            "rsi":             safe("RSI"),
            "rsi_zone":        latest.get("RSI_Zone", ""),
            "ma_20":           safe("MA_20"),
            "ma_50":           safe("MA_50"),
            "ma_200":          safe("MA_200"),
            "cross_type":      latest.get("Cross_Type", ""),
            "long_term_trend": latest.get("Long_Term_Trend", ""),
            "obv_rising":      latest.get("OBV_Rising", ""),
            "bb_pct":          safe("BB_Pct"),
            "volume_ratio":    safe("Volume_Ratio"),
            "signal_details":  " | ".join(details),
        })
        print(f"  {symbol:12s} | Score: {score:+4d} | {label}")
    except Exception as e:
        print(f"  {symbol:12s} | ERROR: {e}")

signals_df = pd.DataFrame(signals_rows).sort_values("score", ascending=False)
signals_df.reset_index(drop=True, inplace=True)

# Print grouped summary
for lbl in ["STRONG BUY", "BUY", "NEUTRAL/HOLD", "SELL", "STRONG SELL"]:
    sub = signals_df[signals_df["signal"] == lbl]
    if len(sub):
        print(f"\n  -- {lbl} ({len(sub)}) --")
        for _, r in sub.iterrows():
            if lbl in ("BUY", "STRONG BUY"):
                print(f"    {r['symbol']:12s}  ₹{r['close']:>10.2f}  "
                      f"SL {r['stop_loss']} ({r['sl_pct']})  "
                      f"TP {r['target']} ({r['upside_pct']})  "
                      f"Qty:{r['qty']}  {r['position_value']}  "
                      f"RSI {r['rsi']:.1f}")
            else:
                print(f"    {r['symbol']:12s}  ₹{r['close']:>10.2f}  RSI {r['rsi']:.1f}")


# ==============================================================================
# STEP 5A: ARIMA FORECASTS
# ==============================================================================

print("\n[STEP 5A] ARIMA Forecasts (close price only — univariate)...")
print("="*80)

ml_symbols = ML_SYMBOLS if ML_SYMBOLS else list(df_ind["symbol"].unique())
arima_rows = []

for symbol in ml_symbols:
    s_df  = df_ind[df_ind["symbol"] == symbol].copy()
    close = s_df["close"]
    print(f"  {symbol:12s}", end="  ")
    t1 = time.time()
    fc, ci = ARIMAForecaster.forecast(close)
    if fc is not None:
        row = {"symbol": symbol}
        for d in range(1, ML_FORECAST_DAYS + 1):
            row[f"arima_d{d}"]    = round(float(fc.values[d-1]), 2)
            row[f"arima_d{d}_lo"] = round(float(ci[d-1][0]), 2)
            row[f"arima_d{d}_hi"] = round(float(ci[d-1][1]), 2)
        arima_rows.append(row)
        print(f"done ({time.time()-t1:.1f}s)  D+1: ₹{fc.values[0]:.2f}")
    else:
        print("SKIPPED (insufficient data)")

arima_df = pd.DataFrame(arima_rows)
print(f"\n  ARIMA completed for {len(arima_df)} symbols.")


# ==============================================================================
# STEP 5B: PROPHET FORECASTS
# ==============================================================================

print("\n[STEP 5B] Prophet Forecasts (close + RSI + Volume_Ratio regressors)...")
print("="*80)

prophet_rows = []

for symbol in ml_symbols:
    s_df = df_ind[df_ind["symbol"] == symbol].copy()
    print(f"  {symbol:12s}", end="  ")
    t1  = time.time()
    fwd = ProphetForecaster.forecast(s_df)
    if fwd is not None:
        row  = {"symbol": symbol}
        vals = fwd["yhat"].values
        lo   = fwd["yhat_lower"].values
        hi   = fwd["yhat_upper"].values
        for d in range(1, min(ML_FORECAST_DAYS, len(vals)) + 1):
            row[f"prophet_d{d}"]    = round(float(vals[d-1]), 2)
            row[f"prophet_d{d}_lo"] = round(float(lo[d-1]),   2)
            row[f"prophet_d{d}_hi"] = round(float(hi[d-1]),   2)
        prophet_rows.append(row)
        print(f"done ({time.time()-t1:.1f}s)  D+1: ₹{vals[0]:.2f}")
    else:
        print("SKIPPED (insufficient data)")

prophet_df = pd.DataFrame(prophet_rows)
print(f"\n  Prophet completed for {len(prophet_df)} symbols.")


# ==============================================================================
# STEP 5C: XGBOOST FORECASTS  (replaces LSTM)
# ==============================================================================

print("\n[STEP 5C] XGBoost Forecasts (close + 12 indicator features)...")
print("="*80)
print(f"  RETRAIN_XGB = {RETRAIN_XGB}")
print(f"  Models dir  : {ML_MODELS_DIR}")
print(f"  Lag days    : {XGB_LAG_DAYS} (last 20 days of each indicator)")
print()

xgb_rows  = []
xgb_metas = []

for symbol in ml_symbols:
    s_df  = df_ind[df_ind["symbol"] == symbol].copy()
    print(f"  {symbol:12s}", end="  ")
    t1    = time.time()
    model = XGBoostForecaster(symbol)
    meta  = model.train(s_df)

    if meta is None:
        print("SKIPPED (insufficient data)")
        continue

    forecasts = model.predict_next(s_df)
    if forecasts:
        row = {"symbol": symbol}
        for d, v in enumerate(forecasts, 1):
            row[f"xgb_d{d}"] = v
        xgb_rows.append(row)
        xgb_metas.append(meta)
        cached = "cached" if (os.path.exists(model.model_path) and not RETRAIN_XGB) else "trained"
        print(f"{cached} ({time.time()-t1:.1f}s)  "
              f"MAE={meta['mae']:.2f}  RMSE={meta['rmse']:.2f}  MAPE={meta['mape']:.2f}%  "
              f"D+1: ₹{forecasts[0]:.2f}")
    else:
        print("SKIPPED (predict failed)")

xgb_df      = pd.DataFrame(xgb_rows)
xgb_meta_df = pd.DataFrame(xgb_metas)
print(f"\n  XGBoost completed for {len(xgb_df)} symbols.")
if len(xgb_meta_df):
    print(f"  Avg MAPE : {xgb_meta_df['mape'].mean():.2f}%")
    print(f"  Avg MAE  : {xgb_meta_df['mae'].mean():.2f}")


# ==============================================================================
# STEP 6: MERGE ALL + CONSENSUS
# ==============================================================================

print("\n[STEP 6] Merging signals + ML forecasts...")
print("="*80)

final_df = signals_df.copy()
if len(arima_df)   > 0: final_df = final_df.merge(arima_df,   on="symbol", how="left")
if len(prophet_df) > 0: final_df = final_df.merge(prophet_df, on="symbol", how="left")
if len(xgb_df)     > 0: final_df = final_df.merge(xgb_df,     on="symbol", how="left")

# ── 3-model MEDIAN consensus (odd number = no tie) ──────────────────────────
# Median is robust: if one model has a bad day, the middle value wins
d1_cols = [c for c in ("arima_d1", "prophet_d1", "xgb_d1") if c in final_df.columns]
if d1_cols:
    final_df["ml_consensus_d1"] = final_df[d1_cols].median(axis=1).round(2)
    final_df["ml_upside_pct"]   = (
        (final_df["ml_consensus_d1"] - final_df["close"])
        / final_df["close"] * 100
    ).round(2)
    final_df["models_used"]     = len(d1_cols)

# ── Swing verdict for BUY signals ───────────────────────────────────────────
if "ml_consensus_d1" in final_df.columns:
    def swing_verdict(row):
        if row["signal"] not in ("BUY", "STRONG BUY"):
            return ""
        tp = row.get("target", np.nan)
        sl = row.get("stop_loss", np.nan)
        ml = row.get("ml_consensus_d1", np.nan)
        if any(pd.isna(x) for x in [tp, sl, ml]):
            return "ML data missing"
        try:
            tp = float(tp); sl = float(sl)
        except:
            return "invalid SL/TP"
        if ml >= tp:   return f"ML confirms TP ₹{tp} likely by D+1"
        elif ml <= sl: return f"ML warns SL ₹{sl} at risk"
        else:
            pct = round((ml - row["close"]) / row["close"] * 100, 2)
            return f"ML: +{pct}% toward TP. Hold."
    final_df["ml_swing_verdict"] = final_df.apply(swing_verdict, axis=1)

# Save
final_df.to_csv(SIGNALS_CSV, index=False)

ml_parts = [f for f in [arima_df, prophet_df, xgb_df] if len(f) > 0]
if ml_parts:
    ml_out = ml_parts[0]
    for f in ml_parts[1:]:
        ml_out = ml_out.merge(f, on="symbol", how="outer")
    ml_out.to_csv(ML_FORECASTS_CSV, index=False)

print(f"  Saved: {SIGNALS_CSV}")
print(f"  Saved: {ML_FORECASTS_CSV}")
print(f"  Rows : {len(final_df)}")

# Display BUY signals with ML confirmation
buy_signals = final_df[final_df["signal"].isin(["BUY", "STRONG BUY"])]
if len(buy_signals):
    print(f"\n  BUY signals with ML consensus:")
    for _, r in buy_signals.iterrows():
        print(f"    {r['symbol']:12s}  ₹{r['close']:>10.2f}  "
              f"→ ML D+1: ₹{r.get('ml_consensus_d1','N/A')}  "
              f"({r.get('ml_upside_pct','N/A')}%)  "
              f"{r.get('ml_swing_verdict','')}")


# ==============================================================================
# STEP 7: BACKTESTING
# Your rules: BUY signal → hold max 40 days → TP (+BB_width) or SL (-BB_width/2)
# Capital: ₹20,000 per trade | Max 2 simultaneous trades
# ==============================================================================

print("\n[STEP 7] Backtesting signals (last 6 months)...")
print("="*80)

def run_backtest(df_ind, lookback_days=180, hold_days=40):
    """
    Replay historical signals and check if BUY calls made money.
    Uses your exact rules: SL = BB_Width/2, TP = BB_Width, RR=1:2
    """
    results    = []
    cutoff     = df_ind["date"].max() - pd.Timedelta(days=lookback_days)
    all_syms   = df_ind["symbol"].unique()

    for symbol in all_syms:
        s_df = df_ind[df_ind["symbol"] == symbol].copy().reset_index(drop=True)
        hist = s_df[s_df["date"] <= cutoff].reset_index(drop=True)

        for i in range(len(hist) - hold_days):
            row = hist.iloc[i]
            _, score, label, _ = TradingSignals.generate(hist.iloc[:i+1])
            if label not in ("BUY", "STRONG BUY"):
                continue

            entry     = row["close"]
            bb_upper  = row.get("BB_Upper", np.nan)
            bb_lower  = row.get("BB_Lower", np.nan)

            if not any(pd.isna(x) for x in [bb_upper, bb_lower]):
                bb_width = bb_upper - bb_lower
                sl       = entry - bb_width / 2
                tp       = entry + bb_width
            else:
                sl = entry * 0.95
                tp = entry * 1.10

            future  = s_df.iloc[i+1 : i+1+hold_days]["close"]
            outcome = "HOLD (timeout)"
            days_held = hold_days

            for j, fp in enumerate(future, 1):
                if fp >= tp:
                    outcome = "WIN"; days_held = j; break
                if fp <= sl:
                    outcome = "LOSS"; days_held = j; break

            qty    = int((CAPITAL * CAPITAL_PER_TRADE) / entry) if entry > 0 else 0
            pnl    = qty * (future.iloc[days_held-1] - entry) if len(future) >= days_held else 0

            results.append({
                "symbol":    symbol,
                "date":      row["date"],
                "signal":    label,
                "score":     score,
                "entry":     round(entry, 2),
                "sl":        round(sl, 2),
                "tp":        round(tp, 2),
                "outcome":   outcome,
                "days_held": days_held,
                "qty":       qty,
                "pnl":       round(pnl, 2),
            })

    if not results:
        print("  No BUY signals found in lookback period.")
        return pd.DataFrame()

    bt_df     = pd.DataFrame(results)
    total     = len(bt_df)
    wins      = (bt_df["outcome"] == "WIN").sum()
    losses    = (bt_df["outcome"] == "LOSS").sum()
    win_rate  = wins / total * 100
    total_pnl = bt_df["pnl"].sum()

    print(f"  Backtest period  : last {lookback_days} calendar days")
    print(f"  Total BUY signals: {total}")
    print(f"  Wins  : {wins}  ({win_rate:.1f}%)")
    print(f"  Losses: {losses}  ({100-win_rate:.1f}%)")
    print(f"  Timeout (held full): {total - wins - losses}")
    print(f"  Total P&L: ₹{total_pnl:,.2f}")
    print(f"  Avg P&L per trade: ₹{bt_df['pnl'].mean():,.2f}")
    print(f"\n  Signal score distribution:")
    for lbl in ["STRONG BUY", "BUY"]:
        sub = bt_df[bt_df["signal"] == lbl]
        if len(sub):
            w = (sub["outcome"] == "WIN").sum()
            print(f"    {lbl:12s}: {len(sub)} trades | {w}/{len(sub)} wins "
                  f"({w/len(sub)*100:.1f}%)  | P&L: ₹{sub['pnl'].sum():,.2f}")

    bt_df.to_csv(BACKTEST_CSV, index=False)
    print(f"\n  Saved: {BACKTEST_CSV}")
    return bt_df

backtest_df = run_backtest(df_ind)


# ==============================================================================
# STEP 8: FINAL SUMMARY
# ==============================================================================

print(f"\n{'='*80}")
print("  PIPELINE COMPLETE")
print(f"{'='*80}")

for lbl in ["STRONG BUY", "BUY", "NEUTRAL/HOLD", "SELL", "STRONG SELL"]:
    sub = final_df[final_df["signal"] == lbl]
    print(f"  {lbl:15s}: {len(sub):2d}")

print()
print("  Files created:")
print(f"    1. {INDICATORS_CSV}")
print(f"    2. {SIGNALS_CSV}  ← main output with ML consensus")
print(f"    3. {ML_FORECASTS_CSV}")
print(f"    4. {BACKTEST_CSV}")
print(f"    5. {ML_MODELS_DIR}/  ← cached XGBoost .json models")
print()
print("  Model summary:")
print(f"    ARIMA   : univariate close price → trend extrapolation")
print(f"    Prophet : close + RSI + Volume_Ratio → seasonality aware")
print(f"    XGBoost : close + 12 indicators × 20 lags → bullish pattern learning")
print(f"    Consensus: MEDIAN of 3 forecasts (odd=no tie, robust to 1 bad model)")
print()
print("  Next step: streamlit run 03_streamlit_app.py")
print(f"{'='*80}")
print(f"  Finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'='*80}")
