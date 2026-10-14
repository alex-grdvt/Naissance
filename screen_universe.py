"""screen_universe.py — S&P 500 screen (work in progress).

Intended to run the Naissance backtest over every current S&P 500 constituent
and rank the results by total return. Constituent scraping, resumable CSV
output and ranking are implemented. run_backtest_for_ticker is a placeholder
and raises NotImplementedError until it is connected to the backtest engine.

No orders are placed; this is research code only.

Ranking by total return alone favours volatile names with one large run in the
window. The per-ticker CSV therefore also records Sharpe ratio and maximum
drawdown, so that results can be re-sorted without rerunning.

Usage (once run_backtest_for_ticker is implemented):
    python screen_universe.py --years 4 --out sp500_screen.csv
    python screen_universe.py --years 4 --out sp500_screen.csv --limit 25
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import time
from dataclasses import dataclass, fields
from datetime import date
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("naissance.screen_universe")

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class TickerResult:
    """Backtest metrics for one ticker; error is set when the backtest failed."""
    ticker: str
    company: str
    total_return_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    num_trades: Optional[int] = None
    error: Optional[str] = None


def get_sp500_constituents() -> list[tuple[str, str]]:
    """Scrapes the current S&P 500 list (ticker, company name) from Wikipedia.

    The list changes over time, so runs several months apart screen slightly
    different universes. Requires pandas and lxml.
    """
    import pandas as pd

    resp = requests.get(WIKIPEDIA_SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]  # first table on the page is the constituent list
    # Wikipedia writes share classes with a dot (BRK.B); Yahoo Finance expects a dash
    # (BRK-B).
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    names = df["Security"].tolist()
    return list(zip(tickers, names))


def run_backtest_for_ticker(ticker: str, start: date, end: date) -> TickerResult:
    """Placeholder: should backtest one ticker between start and end and return a
    TickerResult with at least total_return_pct. Not yet connected to the
    backtest engine.
    """
    raise NotImplementedError(
        "run_backtest_for_ticker is not yet connected to the backtest engine."
    )


def load_already_done(csv_path: str) -> set[str]:
    """Returns the set of tickers already recorded in csv_path, so that a rerun can skip them."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["ticker"] for row in csv.DictReader(f)}


def append_result(csv_path: str, result: TickerResult) -> None:
    """Appends one result row to csv_path, writing the header if the file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(TickerResult)])
        if not file_exists:
            writer.writeheader()
        writer.writerow(result.__dict__)


def main() -> None:
    """Parses arguments, screens each constituent (resuming from the CSV) and prints the top
    results by total return.
    """
    parser = argparse.ArgumentParser(description="Screen the S&P 500 through Naissance's backtest engine")
    parser.add_argument("--years", type=int, default=4, help="Trailing years of history to backtest (default 4)")
    parser.add_argument("--out", type=str, default="sp500_screen.csv", help="CSV file to write/resume from")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tickers (useful for a test pass)")
    parser.add_argument("--top", type=int, default=15, help="How many top results to print at the end")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between tickers, for rate-limited data sources")
    args = parser.parse_args()

    end = date.today()
    start = end.replace(year=end.year - args.years)

    log.info("Fetching S&P 500 constituent list...")
    constituents = get_sp500_constituents()
    if args.limit:
        constituents = constituents[: args.limit]
    log.info("Screening %d tickers from %s to %s", len(constituents), start, end)

    already_done = load_already_done(args.out)
    if already_done:
        log.info("Resuming: %d tickers already in %s, skipping those.", len(already_done), args.out)

    for i, (ticker, company) in enumerate(constituents, 1):
        if ticker in already_done:
            continue
        log.info("[%d/%d] %s (%s)", i, len(constituents), ticker, company)
        try:
            result = run_backtest_for_ticker(ticker, start, end)
            result.company = company
        except Exception as e:
            log.warning("  failed: %s", e)
            result = TickerResult(ticker=ticker, company=company, error=str(e))
        append_result(args.out, result)
        if args.sleep:
            time.sleep(args.sleep)

    # Rank
    rows = []
    with open(args.out, newline="") as f:
        rows = list(csv.DictReader(f))
    scored = [r for r in rows if r.get("total_return_pct") not in (None, "", "None")]
    scored.sort(key=lambda r: float(r["total_return_pct"]), reverse=True)

    print(f"\nTop {args.top} by total return over the trailing {args.years} years:\n")
    for r in scored[: args.top]:
        print(f"  {r['ticker']:<6} {r['company'][:35]:<35} "
              f"return={float(r['total_return_pct']):+.1f}%  "
              f"sharpe={r.get('sharpe', 'n/a')}  "
              f"max_dd={r.get('max_drawdown_pct', 'n/a')}")

    failed = [r for r in rows if r.get("error")]
    if failed:
        print(f"\n{len(failed)} tickers failed (see 'error' column in {args.out}).")


if __name__ == "__main__":
    main()
