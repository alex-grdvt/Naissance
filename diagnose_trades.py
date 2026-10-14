"""
diagnose_trades.py — trade-level diagnostics for Naissance backtest runs.

Reads closed trades from `trade_log` in naissance.db, computes per-trade,
per-regime and per-year return stats, and writes the results back to the
database as a permanent, queryable record (mirrors the `backtest_results`
pattern used for whole-backtest summaries).

Tables created if not already present:
  diagnostics_runs   - one row per script run
  trade_diagnostics  - one row per trade
  regime_diagnostics - per-regime aggregates for a run
  yearly_diagnostics - per-year aggregates for a run
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from portfolio_state import DB_PATH



def ensure_schema(conn):
    """Create the diagnostics tables if they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS diagnostics_runs (
            run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts      TEXT NOT NULL,
            n_trades    INTEGER NOT NULL,
            source      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_diagnostics (
            run_id      INTEGER NOT NULL,
            trade_id    INTEGER,
            ticker      TEXT,
            regime      TEXT,
            entry_date  TEXT,
            exit_date   TEXT,
            entry_year  INTEGER,
            return_pct  REAL,
            FOREIGN KEY (run_id) REFERENCES diagnostics_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS regime_diagnostics (
            run_id      INTEGER NOT NULL,
            regime      TEXT,
            n_trades    INTEGER,
            mean_return REAL,
            sum_return  REAL,
            min_return  REAL,
            max_return  REAL,
            FOREIGN KEY (run_id) REFERENCES diagnostics_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS yearly_diagnostics (
            run_id      INTEGER NOT NULL,
            entry_year  INTEGER,
            n_trades    INTEGER,
            mean_return REAL,
            sum_return  REAL,
            FOREIGN KEY (run_id) REFERENCES diagnostics_runs(run_id)
        );
        """
    )
    conn.commit()


def load_trades(conn, source="backtest"):
    """Load all closed trades for a given source, with return_pct, entry_year and days_held
    added.
    """
    df = pd.read_sql_query(
        """
        SELECT
            id AS trade_id,
            ticker,
            position,
            size,
            entry_price,
            exit_price,
            entry_date,
            exit_date,
            regime
        FROM trade_log
        WHERE source = ?
        ORDER BY entry_date
        """,
        conn,
        params=(source,),
    )

    if df.empty:
        return df

    df["return_pct"] = (
        (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
    )
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["entry_year"] = df["entry_date"].dt.year
    df["days_held"] = (df["exit_date"] - df["entry_date"]).dt.days
    return df


def summarize(df):
    """Compute headline stats and regime/year aggregates for a trade set."""
    summary = {
        "n_trades": len(df),
        "mean_return": df["return_pct"].mean(),
        "best_trade": df["return_pct"].max(),
        "worst_trade": df["return_pct"].min(),
        "mean_size_pct": df["size"].mean() * 100,
        "max_size_pct": df["size"].max() * 100,
    }
    worst_10 = df.sort_values("return_pct").head(10)
    by_regime = df.groupby("regime")["return_pct"].agg(
        ["count", "mean", "sum", "min", "max"]
    )
    by_year = df.groupby("entry_year")["return_pct"].agg(["count", "mean", "sum"])
    return summary, worst_10, by_regime, by_year


def save_run(conn, df, by_regime, by_year, source="backtest"):
    """Persist a diagnostics run and its aggregates; returns the new run_id."""
    run_ts = datetime.now(timezone.utc).isoformat()

    cur = conn.execute(
        "INSERT INTO diagnostics_runs (run_ts, n_trades, source) VALUES (?, ?, ?)",
        (run_ts, len(df), source),
    )
    run_id = cur.lastrowid

    trades_out = df[
        ["trade_id", "ticker", "regime", "entry_date", "exit_date", "entry_year", "return_pct"]
    ].copy()
    trades_out.insert(0, "run_id", run_id)
    trades_out["entry_date"] = trades_out["entry_date"].dt.strftime("%Y-%m-%d")
    trades_out["exit_date"] = trades_out["exit_date"].dt.strftime("%Y-%m-%d")
    trades_out.to_sql("trade_diagnostics", conn, if_exists="append", index=False)

    regime_out = by_regime.reset_index().rename(
        columns={"count": "n_trades", "mean": "mean_return", "sum": "sum_return",
                 "min": "min_return", "max": "max_return"}
    )
    regime_out.insert(0, "run_id", run_id)
    regime_out.to_sql("regime_diagnostics", conn, if_exists="append", index=False)

    yearly_out = by_year.reset_index().rename(
        columns={"count": "n_trades", "mean": "mean_return", "sum": "sum_return"}
    )
    yearly_out.insert(0, "run_id", run_id)
    yearly_out.to_sql("yearly_diagnostics", conn, if_exists="append", index=False)

    conn.commit()
    return run_id


def main(source="backtest"):
    """Computes diagnostics for the closed trades of the given source, saves them to the
    database and prints a summary.
    """
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    df = load_trades(conn, source=source)
    if df.empty:
        print(f"No trades found with source='{source}'.")
        conn.close()
        return

    summary, worst_10, by_regime, by_year = summarize(df)
    run_id = save_run(conn, df, by_regime, by_year, source=source)

    print(f"\nSaved diagnostics run_id={run_id} ({summary['n_trades']} trades) to {DB_PATH}")
    print("=" * 80)
    print(f"Average return: {summary['mean_return']:.2f}%")
    print(f"Best trade:     {summary['best_trade']:.2f}%")
    print(f"Worst trade:    {summary['worst_trade']:.2f}%")

    print("\nWORST 10 TRADES")
    print("=" * 80)
    print(worst_10[["ticker", "entry_date", "exit_date", "regime", "return_pct"]].to_string(index=False))

    print("\nREGIME PERFORMANCE")
    print("=" * 80)
    print(by_regime)

    print("\nYEARLY PERFORMANCE")
    print("=" * 80)
    print(by_year)

    print("\nALL TRADES")
    print("=" * 80)
    print(
        df[["ticker", "entry_date", "exit_date", "regime", "return_pct"]]
        .to_string(index=False)
    )

    print("\nDAYS HELD")
    print("=" * 80)
    print(df["days_held"].describe())

    print(f"\nAverage position size: {summary['mean_size_pct']:.2f}%")
    print(f"Max position size:     {summary['max_size_pct']:.2f}%")

    print(
        f"\nQuery this run later with:\n"
        f"  SELECT * FROM trade_diagnostics WHERE run_id = {run_id};\n"
        f"  SELECT * FROM regime_diagnostics WHERE run_id = {run_id};\n"
        f"  SELECT * FROM yearly_diagnostics WHERE run_id = {run_id};"
    )

    conn.close()


if __name__ == "__main__":
    main()