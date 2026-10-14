"""Backtests a fixed list of tickers one at a time and ranks them by Sharpe ratio.

Each ticker is run through run_full_backtest.py in a separate process with
sentiment disabled (--fast). The latest PORTFOLIO row in backtest_results is
then read back, and the combined table is written to ticker_sweep_results.csv.

Usage:
    python ticker_sweep.py
"""

import os
import sqlite3
import subprocess
import sys

import pandas as pd

from portfolio_state import DB_PATH

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "ticker_sweep_results.csv")

TICKERS = [
    "MSFT", "AAPL", "NVDA", "AVGO", "AMD",
    "JPM", "GS", "MS", "BLK",
    "LLY", "UNH", "ABBV", "ISRG",
    "CAT", "GE", "RTX", "UBER",
    "WMT", "COST", "PG", "KO",
    "AMZN", "TSLA", "HD", "BKNG",
    "XOM", "CVX", "SLB",
    "LIN", "FCX", "NEM",
    "META", "GOOGL", "NFLX",
    "PLTR", "SHOP", "MELI",
    "SPGI", "ICE",
]


def run_ticker(ticker):
    """Backtests one ticker and returns its portfolio-level metrics, or None if no result was
    stored.
    """
    subprocess.run(
        [sys.executable, "run_full_backtest.py", f"--tickers={ticker}", "--fast"],
        cwd=PROJECT_ROOT,
        check=False,
    )

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM backtest_results
            WHERE ticker = 'PORTFOLIO'
            ORDER BY rowid DESC
            LIMIT 1
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "Ticker": ticker,
        "Sharpe": row["sharpe"],
        "MaxDD": row["max_drawdown"],
        "TotalReturn": row["total_return_pct"],
        "CAGR": row["annualized_return_pct"],
    }


def main():
    """Runs the sweep, writes the ranked table to CSV and prints it."""
    results = []
    for ticker in TICKERS:
        print(f"\nRunning {ticker}...")
        result = run_ticker(ticker)
        if result is None:
            print(f"No results found for {ticker}")
            continue
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="Sharpe", ascending=False)

    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\nFinal rankings:")
    print(results_df)


if __name__ == "__main__":
    main()
