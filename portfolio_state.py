"""SQLite persistence layer for the pipeline: schema definitions and CRUD helpers
for portfolio state, trade history, regime history, signal history, price
cache, sentiment history, system maturity, backtest results and portfolio
allocations. Read and written throughout main.py and the layers (backtest_engine,
position_sizing, asset_sizing and others) through the functions below rather
than direct SQL elsewhere in the codebase.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'naissance.db')


@contextmanager
def get_connection():
    """
    Yields a sqlite3 connection and ensures it is closed on exit, including
    when the caller raises. timeout=30 makes concurrent access wait for a
    lock rather than failing immediately; WAL allows readers and a writer to
    coexist without blocking each other.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
    finally:
        conn.close()


def initialise_db():
    """Creates all pipeline tables if they do not already exist and applies schema migrations
    for existing databases.
    """
    with get_connection() as conn:
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_state (
                ticker TEXT PRIMARY KEY,
                position TEXT,          -- 'long', 'short', 'flat'
                entry_price REAL,
                entry_date TEXT,
                current_regime TEXT,
                previous_regime TEXT,
                stop_loss REAL,
                position_size REAL DEFAULT 0,
                updated_at TEXT
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                regime TEXT,
                previous_regime TEXT,
                timestamp TEXT
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                action TEXT,            -- 'BUY', 'SHORT', 'EXIT'
                position TEXT,          -- 'long', 'short'
                entry_price REAL,
                exit_price REAL,
                entry_date TEXT,
                exit_date TEXT,
                pnl REAL,
                regime TEXT,
                alpha_score REAL,
                source TEXT DEFAULT 'live',  -- 'backtest', 'paper', 'live'
                notional REAL,          -- $ actually deployed on this trade (None if the caller did not pass it)
                dollar_pnl REAL         -- $ P&L on this trade = notional * net_pnl_pct/100 (None if the caller did not pass it)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                alpha_score REAL,
                raw_score REAL,
                factor_alpha REAL,
                regime TEXT,
                macro_mode TEXT
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS price_cache (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                score REAL,
                label TEXT,
                article_count INTEGER,
                UNIQUE (ticker, date)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS system_maturity (
                ticker TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                sentiment_active INTEGER DEFAULT 0,  -- 0/1 bool
                kelly_mode TEXT DEFAULT 'default'    -- 'default', 'backtest', 'live'
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                run_date TEXT,
                n_trades INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                sharpe REAL,
                max_drawdown REAL,
                total_return_pct REAL,
                annualized_return_pct REAL,
                source TEXT DEFAULT 'backtest'
            )
        ''')

        # One row per ticker per sizing run, recording the allocation decision itself
        # rather than the trade that resulted from it. Written by
        # asset_sizing.run_multi_asset.
        c.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT,
                ticker TEXT,
                weight REAL,              -- final fraction of portfolio_value, post drawdown/vol throttle
                raw_weight REAL,          -- pre-throttle weight from allocate_portfolio
                edge REAL,
                score_gap REAL,
                regime TEXT,
                category TEXT,
                asset_class TEXT,
                capped_by TEXT,           -- 'kelly' | 'ticker_cap' | 'group_cap' | NULL
                cash_residual REAL        -- same value repeated on every row of a run,
                                          -- for convenience querying a single run_date
            )
        ''')

        # Migration: add total_return_pct / annualized_return_pct to
        # backtest_results if missing (existing DBs).
        existing_br_cols = [row[1] for row in c.execute('PRAGMA table_info(backtest_results)').fetchall()]
        if 'total_return_pct' not in existing_br_cols:
            c.execute("ALTER TABLE backtest_results ADD COLUMN total_return_pct REAL")
        if 'annualized_return_pct' not in existing_br_cols:
            c.execute("ALTER TABLE backtest_results ADD COLUMN annualized_return_pct REAL")

        # Migration: add source column to trade_log if missing (existing DBs)
        existing_cols = [row[1] for row in c.execute('PRAGMA table_info(trade_log)').fetchall()]
        if 'source' not in existing_cols:
            c.execute("ALTER TABLE trade_log ADD COLUMN source TEXT DEFAULT 'live'")

        # Migration: add notional/dollar_pnl columns to trade_log if missing
        # (existing DBs). Historical rows read NULL for these columns.
        if 'notional' not in existing_cols:
            c.execute("ALTER TABLE trade_log ADD COLUMN notional REAL")
        if 'dollar_pnl' not in existing_cols:
            c.execute("ALTER TABLE trade_log ADD COLUMN dollar_pnl REAL")

        # Migration: add position_size column to portfolio_state if missing (existing DBs)
        existing_ps_cols = [row[1] for row in c.execute('PRAGMA table_info(portfolio_state)').fetchall()]
        if 'position_size' not in existing_ps_cols:
            c.execute("ALTER TABLE portfolio_state ADD COLUMN position_size REAL DEFAULT 0")

        conn.commit()


def get_state(ticker):
    """Returns the portfolio_state row for ticker as a dict, or safe flat/zero defaults if
    none exists.
    """
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM portfolio_state WHERE ticker = ?', (ticker,))
        # actual column order on disk; differs between a fresh DB (CREATE
        # TABLE order) and a migrated one (ALTER TABLE appends at the end)
        cols = [d[0] for d in c.description]
        row = c.fetchone()

    if row is None:
        return {
            'ticker': ticker,
            'position': 'flat',
            'entry_price': None,
            'entry_date': None,
            'current_regime': None,
            'previous_regime': None,
            'stop_loss': None,
            'position_size': 0,
            'updated_at': None
        }

    return dict(zip(cols, row))


def update_state(ticker, **kwargs):
    """Merges kwargs into the current state for ticker, stamps updated_at, and upserts the
    row into portfolio_state.
    """
    state = get_state(ticker)
    state.update(kwargs)
    state['updated_at'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    with get_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO portfolio_state
            (ticker, position, entry_price, entry_date, current_regime, previous_regime, stop_loss, position_size, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            state['position'],
            state['entry_price'],
            state['entry_date'],
            state['current_regime'],
            state['previous_regime'],
            state['stop_loss'],
            state.get('position_size', 0),
            state['updated_at']
        ))
        conn.commit()


def log_regime_change(ticker, new_regime, previous_regime):
    """Appends a row to regime_history when new_regime differs from previous_regime; a no-op
    otherwise.
    """
    if new_regime == previous_regime:
        return

    with get_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO regime_history (ticker, regime, previous_regime, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (ticker, new_regime, previous_regime, datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()


def log_trade(ticker, action, position, entry_price, exit_price,
              entry_date, regime, alpha_score, source='live', exit_date=None,
              notional=None, dollar_pnl=None):
    """
    Logs a trade row to trade_log and refreshes the ticker's maturity state.
    notional and dollar_pnl are optional; callers that do not track per-trade
    dollar sizing omit them and NULL is logged instead. pnl is the raw
    price-return percentage, independent of position size, and does not
    reflect capital actually risked; code needing the real dollar scale of a
    trade should read notional/dollar_pnl rather than inferring it from pnl.
    """
    pnl = None
    if entry_price and exit_price:
        if position == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
        elif position == 'short':
            pnl = (entry_price - exit_price) / entry_price * 100

    if exit_date is None:
        exit_date = datetime.today().strftime('%Y-%m-%d')

    with get_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO trade_log
            (ticker, action, position, entry_price, exit_price,
             entry_date, exit_date, pnl, regime, alpha_score, source,
             notional, dollar_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, action, position, entry_price, exit_price,
            entry_date, exit_date,
            pnl, regime, alpha_score, source,
            notional, dollar_pnl
        ))
        conn.commit()
    refresh_maturity(ticker)


def refresh_maturity(ticker):
    """Recomputes total_trades and kelly_mode for ticker from trade_log counts and persists
    them via update_maturity.
    """
    with get_connection() as conn:
        c = conn.cursor()

        c.execute("""
            SELECT COUNT(*)
            FROM trade_log
            WHERE ticker = ?
            AND source = 'backtest'
            AND pnl IS NOT NULL
        """, (ticker,))
        backtest_trades = c.fetchone()[0]

        c.execute("""
            SELECT COUNT(*)
            FROM trade_log
            WHERE ticker = ?
            AND source = 'live'
            AND pnl IS NOT NULL
        """, (ticker,))
        live_trades = c.fetchone()[0]

    total_trades = backtest_trades + live_trades

    if live_trades >= 10:
        mode = "live"
    elif backtest_trades >= 5:
        mode = "backtest"
    else:
        mode = "default"

    update_maturity(
        ticker=ticker,
        total_trades=total_trades,
        sentiment_active=True,
        kelly_mode=mode,
    )


def log_signal(ticker, alpha_score, raw_score, factor_alpha, regime, macro_mode):
    """Appends a row to signal_history recording the alpha score breakdown for ticker on the
    current date.
    """
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO signal_history (ticker, date, alpha_score, raw_score, factor_alpha, regime, macro_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, datetime.today().strftime('%Y-%m-%d'), alpha_score, raw_score, factor_alpha, regime, macro_mode))
        conn.commit()


def update_regimes(regimes):
    """Updates current_regime for each ticker in regimes, logging any transition to
    regime_history.
    """
    for ticker, new_regime in regimes.items():
        state = get_state(ticker)
        previous_regime = state['current_regime']
        log_regime_change(ticker, new_regime, previous_regime)
        update_state(ticker, current_regime=new_regime, previous_regime=previous_regime)


def get_all_states():
    """Returns the full portfolio_state table as a DataFrame, one row per ticker."""
    with get_connection() as conn:
        df = pd.read_sql('SELECT * FROM portfolio_state', conn)
    return df


def cache_prices(ticker, df):
    """
    Persists a DataFrame of OHLCV rows for ticker into price_cache.
    df must have columns: open, high, low, close, volume (case-insensitive)
    and a DatetimeIndex or 'date' column.
    Existing rows (same ticker+date) are silently ignored via INSERT OR IGNORE.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['ticker'] = ticker

    with get_connection() as conn:
        for _, row in df.iterrows():
            conn.execute('''
                INSERT OR IGNORE INTO price_cache (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['ticker'], row['date'],
                row.get('open'), row.get('high'), row.get('low'),
                row.get('close'), row.get('volume')
            ))
        conn.commit()


def get_cached_prices(ticker, start, end):
    """
    Returns cached OHLCV rows for ticker between start and end (inclusive).
    Returns a DataFrame with columns [date, open, high, low, close, volume],
    or an empty DataFrame if nothing is cached.
    """
    with get_connection() as conn:
        df = pd.read_sql('''
            SELECT date, open, high, low, close, volume
            FROM price_cache
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        ''', conn, params=(ticker, start, end))
    return df


def log_sentiment(ticker, date, score, label, article_count):
    """Persists a sentiment reading. Duplicate (ticker, date) rows are silently ignored."""
    with get_connection() as conn:
        conn.execute('''
            INSERT OR IGNORE INTO sentiment_history (ticker, date, score, label, article_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (ticker, date, score, label, article_count))
        conn.commit()


def get_sentiment_history(ticker, start=None, end=None):
    """Returns sentiment rows for ticker, optionally filtered by date range."""
    with get_connection() as conn:
        if start and end:
            df = pd.read_sql('''
                SELECT date, score, label, article_count
                FROM sentiment_history
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            ''', conn, params=(ticker, start, end))
        else:
            df = pd.read_sql('''
                SELECT date, score, label, article_count
                FROM sentiment_history
                WHERE ticker = ?
                ORDER BY date ASC
            ''', conn, params=(ticker,))
    return df


def update_maturity(ticker, total_trades, sentiment_active, kelly_mode):
    """
    Upserts the maturity row for ticker.
    sentiment_active: bool
    kelly_mode: 'default' | 'backtest' | 'live'
    """
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO system_maturity (ticker, total_trades, sentiment_active, kelly_mode)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                total_trades = excluded.total_trades,
                sentiment_active = excluded.sentiment_active,
                kelly_mode = excluded.kelly_mode
        ''', (ticker, total_trades, int(sentiment_active), kelly_mode))
        conn.commit()


def get_maturity(ticker):
    """Returns the maturity dict for ticker, or safe defaults if not yet set."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM system_maturity WHERE ticker = ?', (ticker,))
        row = c.fetchone()

    if row is None:
        return {
            'ticker': ticker,
            'total_trades': 0,
            'sentiment_active': False,
            'kelly_mode': 'default'
        }

    return {
        'ticker': row[0],
        'total_trades': row[1],
        'sentiment_active': bool(row[2]),
        'kelly_mode': row[3]
    }


def log_backtest_results(ticker, n_trades, win_rate, avg_win, avg_loss,
                         sharpe, max_drawdown, total_return_pct=None,
                         annualized_return_pct=None, source='backtest'):
    """Saves a backtest summary row for ticker."""
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO backtest_results
            (ticker, run_date, n_trades, win_rate, avg_win, avg_loss, sharpe, max_drawdown,
             total_return_pct, annualized_return_pct, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, datetime.today().strftime('%Y-%m-%d'),
            n_trades, win_rate, avg_win, avg_loss, sharpe, max_drawdown,
            total_return_pct, annualized_return_pct, source
        ))
        conn.commit()


def get_backtest_results(ticker):
    """Returns all backtest result rows for ticker, most recent first."""
    with get_connection() as conn:
        df = pd.read_sql('''
            SELECT * FROM backtest_results
            WHERE ticker = ?
            ORDER BY run_date DESC
        ''', conn, params=(ticker,))
    return df


def log_allocation(run_date, ticker, weight, raw_weight, edge, score_gap,
                    regime, category, asset_class, capped_by, cash_residual):
    """Logs one portfolio_allocations row (see its schema in initialise_db). Called
    from asset_sizing.run_multi_asset once per actionable ticker, so a day with N
    allocations writes N rows sharing the same run_date and cash_residual.
    """
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO portfolio_allocations
            (run_date, ticker, weight, raw_weight, edge, score_gap, regime,
             category, asset_class, capped_by, cash_residual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_date, ticker, weight, raw_weight, edge, score_gap, regime,
            category, asset_class, capped_by, cash_residual
        ))
        conn.commit()


def get_allocations(run_date=None, ticker=None):
    """
    Returns portfolio_allocations rows, most recent first. Filter by
    run_date and/or ticker; omit both for the full history.
    """
    query = 'SELECT * FROM portfolio_allocations WHERE 1=1'
    params = []
    if run_date:
        query += ' AND run_date = ?'
        params.append(run_date)
    if ticker:
        query += ' AND ticker = ?'
        params.append(ticker)
    query += ' ORDER BY run_date DESC, weight DESC'

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df


if __name__ == '__main__':
    initialise_db()
    print("DB initialised at:", DB_PATH)