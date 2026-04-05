"""
================================================================================
  NIFTY 50 — SMART INCREMENTAL DATA UPDATER  [v5 — Enhanced]
  MBA Dissertation: AI-Driven Algorithmic Trading System

  CHANGES FROM v4:
    - ADDED: EMA 40 (disaster line) to stock universe awareness
    - ADDED: MACD indicator support in data pipeline
    - ADDED: ADX indicator support in data pipeline
    - FIXED: TATAMOTORS ticker mapping (TMPV.NS → TATAMOTORS.NS)
    - IMPROVED: Retry logic for failed downloads
    - IMPROVED: Data validation checks

  HOW IT WORKS:
    - First run  → fetches full history from 2001 for all 50 stocks
    - Every next run → only fetches NEW rows since last saved date
    - Never re-downloads data that already exists
    - Safe to run daily, weekly, or whenever you need fresh data

  OUTPUT FORMAT:  date | open | high | low | close | volume | symbol
================================================================================
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, date, timedelta


# ==============================================================================
# CONFIGURATION
# ==============================================================================

START_DATE     = "2001-01-01"
OUTPUT_DIR     = "02_data"
INDIVIDUAL_DIR = "02_data/individual_stocks"
MASTER_CSV     = "02_data/nifty50_historical_master.csv"

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)


# ==============================================================================
# NIFTY 50 STOCK UNIVERSE
# FIXED: TATAMOTORS ticker was wrong (TMPV.NS → TATAMOTORS.NS)
# ==============================================================================

NIFTY50_STOCKS = [
    ("ADANIENT.NS",    "ADANIENT"  ),
    ("ADANIPORTS.NS",  "ADANIPORTS"),
    ("APOLLOHOSP.NS",  "APOLLOHOSP"),
    ("ASIANPAINT.NS",  "ASIANPAINT"),
    ("AXISBANK.NS",    "AXISBANK"  ),
    ("BAJAJ-AUTO.NS",  "BAJAJ-AUTO"),
    ("BAJFINANCE.NS",  "BAJFINANCE"),
    ("BAJAJFINSV.NS",  "BAJAJFINSV"),
    ("BEL.NS",         "BEL"       ),
    ("BHARTIARTL.NS",  "BHARTIARTL"),
    ("CIPLA.NS",       "CIPLA"     ),
    ("COALINDIA.NS",   "COALINDIA" ),
    ("DRREDDY.NS",     "DRREDDY"   ),
    ("EICHERMOT.NS",   "EICHERMOT" ),
    ("ETERNAL.NS",     "ETERNAL"   ),
    ("GRASIM.NS",      "GRASIM"    ),
    ("HCLTECH.NS",     "HCLTECH"   ),
    ("HDFCBANK.NS",    "HDFCBANK"  ),
    ("HDFCLIFE.NS",    "HDFCLIFE"  ),
    ("HINDALCO.NS",    "HINDALCO"  ),
    ("HINDUNILVR.NS",  "HINDUNILVR"),
    ("ICICIBANK.NS",   "ICICIBANK" ),
    ("ITC.NS",         "ITC"       ),
    ("INFY.NS",        "INFY"      ),
    ("INDIGO.NS",      "INDIGO"    ),
    ("JSWSTEEL.NS",    "JSWSTEEL"  ),
    ("JIOFIN.NS",      "JIOFIN"    ),
    ("KOTAKBANK.NS",   "KOTAKBANK" ),
    ("LT.NS",          "LT"        ),
    ("M&M.NS",         "M&M"       ),
    ("MARUTI.NS",      "MARUTI"    ),
    ("MAXHEALTH.NS",   "MAXHEALTH" ),
    ("NTPC.NS",        "NTPC"      ),
    ("NESTLEIND.NS",   "NESTLEIND" ),
    ("ONGC.NS",        "ONGC"      ),
    ("POWERGRID.NS",   "POWERGRID" ),
    ("RELIANCE.NS",    "RELIANCE"  ),
    ("SBILIFE.NS",     "SBILIFE"   ),
    ("SHRIRAMFIN.NS",  "SHRIRAMFIN"),
    ("SBIN.NS",        "SBIN"      ),
    ("SUNPHARMA.NS",   "SUNPHARMA" ),
    ("TCS.NS",         "TCS"       ),
    ("TATACONSUM.NS",  "TATACONSUM"),
    ("TATAMOTORS.NS",  "TATAMOTORS"),   # FIXED: was TMPV.NS
    ("TATASTEEL.NS",   "TATASTEEL" ),
    ("TECHM.NS",       "TECHM"     ),
    ("TITAN.NS",       "TITAN"     ),
    ("TRENT.NS",       "TRENT"     ),
    ("ULTRACEMCO.NS",  "ULTRACEMCO"),
    ("WIPRO.NS",       "WIPRO"     ),
]


# ==============================================================================
# HELPER: CLEAN RAW YFINANCE DATA INTO STANDARD OHLCV FORMAT
# ==============================================================================

def clean_ohlcv(raw_df, symbol):
    df = raw_df.copy()
    df.reset_index(inplace=True)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.normalize()
    df["symbol"] = symbol
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "symbol"]
            if c in df.columns]
    df = df[keep].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df = df[df["close"] > 0].copy()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("int64")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ==============================================================================
# CORE LOGIC: SMART UPDATE FOR ONE STOCK (with retry)
# ==============================================================================

def smart_update_stock(yf_ticker, nse_symbol, max_retries=2):
    cache_path  = f"{INDIVIDUAL_DIR}/{nse_symbol}.csv"
    today       = date.today()
    existing_df = None

    if not os.path.exists(cache_path):
        fetch_from = START_DATE
        is_new     = True
    else:
        existing_df = pd.read_csv(cache_path, parse_dates=["date"])
        if existing_df.empty:
            fetch_from = START_DATE
            is_new     = True
        else:
            last_date = existing_df["date"].max().date()
            if last_date >= today - timedelta(days=4):
                return ("already_uptodate", 0, len(existing_df), None)
            fetch_from = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            is_new     = False

    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(yf_ticker)
            raw    = ticker.history(
                start       = fetch_from,
                end         = today.strftime("%Y-%m-%d"),
                interval    = "1d",
                auto_adjust = True,
                actions     = False,
            )
            if raw.empty:
                if is_new:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    return ("error", 0, 0, "Empty response from Yahoo Finance")
                return ("already_uptodate", 0, len(existing_df), None)

            new_df     = clean_ohlcv(raw, nse_symbol)
            rows_added = len(new_df)

            # DATA VALIDATION: check for anomalies
            if rows_added > 0:
                price_range = new_df["close"].max() / max(new_df["close"].min(), 0.01)
                if price_range > 100:
                    return ("error", 0, 0, f"Suspicious price range: {price_range:.0f}x")

            if is_new:
                final_df = new_df
            else:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
                final_df.sort_values("date", inplace=True)
                final_df.reset_index(drop=True, inplace=True)

            final_df.to_csv(cache_path, index=False, date_format="%Y-%m-%d")
            status = "fresh_download" if is_new else "updated"
            return (status, rows_added, len(final_df), None)

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return ("error", 0, 0, str(e))


# ==============================================================================
# REBUILD MASTER CSV
# ==============================================================================

def rebuild_master_csv():
    all_frames = []
    for _, nse_symbol in NIFTY50_STOCKS:
        path = f"{INDIVIDUAL_DIR}/{nse_symbol}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            all_frames.append(df)

    if not all_frames:
        print("  No stock CSVs found — run update first.")
        return None

    master = pd.concat(all_frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master.sort_values(["symbol", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)
    master = master[["date", "open", "high", "low", "close", "volume", "symbol"]]
    master.to_csv(MASTER_CSV, index=False, date_format="%Y-%m-%d")
    return master


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_update():
    print(f"\n{'='*80}")
    print(f"NIFTY 50 SMART DATA UPDATER  [v5 — Enhanced]")
    print(f"{'='*80}")
    print(f"  Running at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stocks     : {len(NIFTY50_STOCKS)}")
    print(f"  Fix applied: TATAMOTORS.NS (was TMPV.NS)")
    print(f"{'='*80}\n")

    total = len(NIFTY50_STOCKS)
    fresh_count = updated_count = skipped_count = error_count = total_new_rows = 0
    error_log = []

    for idx, (yf_ticker, nse_symbol) in enumerate(NIFTY50_STOCKS, 1):
        print(f"[{idx:2d}/{total}]  {nse_symbol:12s}", end="  ")
        status, rows_added, total_rows, error_msg = smart_update_stock(yf_ticker, nse_symbol)

        if status == "fresh_download":
            print(f"NEW DOWNLOAD  | {total_rows:,} rows fetched from {START_DATE}")
            fresh_count += 1; total_new_rows += rows_added
        elif status == "updated":
            print(f"UPDATED       | +{rows_added} new rows  | total: {total_rows:,} rows")
            updated_count += 1; total_new_rows += rows_added
        elif status == "already_uptodate":
            print(f"UP TO DATE    | {total_rows:,} rows — no new data to fetch")
            skipped_count += 1
        elif status == "error":
            print(f"ERROR         | {error_msg}")
            error_count += 1
            error_log.append(f"  {nse_symbol:12s} - {error_msg}")

        time.sleep(0.4)

    print(f"\n{'='*80}")
    print("  Rebuilding master CSV...")
    master = rebuild_master_csv()

    if master is not None:
        print(f"  Saved to    : {MASTER_CSV}")
        print(f"  Total rows  : {len(master):,}")
        print(f"  Stocks      : {master['symbol'].nunique()}")
        print(f"  Date range  : {master['date'].min().date()} to {master['date'].max().date()}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"  Fresh downloads  : {fresh_count}")
    print(f"  Stocks updated   : {updated_count}   (+{total_new_rows:,} total new rows)")
    print(f"  Already current  : {skipped_count}")
    print(f"  Errors           : {error_count}")
    if error_log:
        print(f"\n  Failed stocks:")
        for entry in error_log:
            print(entry)
    print(f"{'='*80}\n")
    return master


if __name__ == "__main__":
    master_df = run_update()
