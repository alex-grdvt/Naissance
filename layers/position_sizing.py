"""Sizes positions with the Kelly criterion plus portfolio-level risk overlays
(drawdown throttling, high-volatility halving, cash and positions-held caps).
Used by main.py's L7 through asset_sizing.py.

Sizing mode (single-ticker all-in versus multi-ticker diversified) is decided in
asset_sizing.py by which function is called. compute_kelly and the drawdown and
consecutive-win helpers here serve the multi-ticker diversified path
(asset_sizing.allocate_portfolio).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from portfolio_state import get_connection, get_maturity

KELLY_CAP = 1.0
KELLY_FRACTION = 3
CASH_FLOOR = 500
MIN_TRADES = 30  # trades needed before a win-rate/b estimate is trusted; below this, compute_kelly uses the default p/b

DRAWDOWN_MILD = 0.08
DRAWDOWN_SEVERE = 0.12
DRAWDOWN_MILD_REDUCE = 0.30
DRAWDOWN_SEVERE_REDUCE = 0.50
CONSECUTIVE_WINS = 2


# Kelly helpers

def compute_kelly(p, b):
    """Returns the fractional Kelly stake for win probability p and win/loss ratio b, capped
    at KELLY_CAP and scaled by KELLY_FRACTION.
    """
    if b <= 0 or p <= 0:
        return 0
    q = 1 - p
    f_full = (p * b - q) / b
    f = f_full * KELLY_FRACTION
    return max(0, min(KELLY_CAP, f))


def get_trade_stats(ticker, source_filter=None):
    """Returns (p, b, n_trades) for ticker from trade_log. source_filter: 'backtest', 'live',
    or None.
    """
    with get_connection() as conn:
        if source_filter:
            df = pd.read_sql(
                'SELECT pnl FROM trade_log WHERE ticker = ? AND source = ? AND pnl IS NOT NULL',
                conn, params=(ticker, source_filter)
            )
        else:
            df = pd.read_sql(
                'SELECT pnl FROM trade_log WHERE ticker = ? AND pnl IS NOT NULL',
                conn, params=(ticker,)
            )

    if len(df) < MIN_TRADES:
        return None

    winners = df[df['pnl'] > 0]['pnl']
    losers = df[df['pnl'] <= 0]['pnl'].abs()
    p = len(winners) / len(df)
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 1
    b = avg_win / avg_loss if avg_loss > 0 else 0

    return p, b, len(df)


# Drawdown

def get_portfolio_drawdown():
    """
    Drawdown as a fraction of peak equity (0.10 = 10% down), built from a
    multiplicative equity curve over live trade_log pnl%, not a raw cumsum.
    Assumes one nominal-size position at a time; does not yet account for
    multiple concurrently-open Kelly-sized positions.
    """
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT pnl FROM trade_log WHERE source = 'live' AND pnl IS NOT NULL ORDER BY id ASC",
            conn
        )

    if df.empty or len(df) < 2:
        return 0.0

    NOMINAL_TRADE_WEIGHT = 0.05  # matches backtest_engine.FIXED_TRADE_SIZE

    weighted_returns = (df['pnl'].values / 100.0) * NOMINAL_TRADE_WEIGHT
    equity = np.cumprod(1 + weighted_returns)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd[-1])


def get_consecutive_wins():
    """Returns the current live winning streak, counting back from the most recent trade
    until a loss is hit.
    """
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT pnl FROM trade_log WHERE pnl IS NOT NULL AND source = 'live' ORDER BY id DESC LIMIT 10",
            conn
        )

    if df.empty:
        return 0

    count = 0
    for pnl in df['pnl']:
        if pnl > 0:
            count += 1
        else:
            break
    return count


# Main sizing

def compute_position_size(ticker, portfolio_value, regime, portfolio_states, high_vol=False):
    """Returns position size in £, sized off Kelly (default/backtest/live per system_maturity)."""
    from layers.backtest_engine import run_backtest
    from data.factor_data import fetch_factor_data

    maturity = get_maturity(ticker)
    kelly_mode = maturity['kelly_mode']

    if kelly_mode == 'default':
        today = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d')
        factor_df = fetch_factor_data(start, today)
        run_backtest(ticker, factor_df)
        maturity = get_maturity(ticker)
        kelly_mode = maturity['kelly_mode']

    if kelly_mode == 'live':
        stats = get_trade_stats(ticker, source_filter='live')
    elif kelly_mode == 'backtest':
        stats = get_trade_stats(ticker, source_filter='backtest')
    else:
        stats = None

    if stats is None:
        p, b = 0.50, 1.20
        print(f'  [SIZE] {ticker} — using default Kelly (p={p}, b={b})')
    else:
        p, b, n = stats
        print(f'  [SIZE] {ticker} — {kelly_mode} stats: p={p:.2f}, b={b:.2f}, n={n}')

    kelly_f = compute_kelly(p, b)
    base_size = portfolio_value * kelly_f

    drawdown = get_portfolio_drawdown()
    consecutive_wins = get_consecutive_wins()

    if drawdown > DRAWDOWN_SEVERE:
        if consecutive_wins >= CONSECUTIVE_WINS:
            reduction = 0
            print(f'  [SIZE] {ticker} — severe drawdown but {consecutive_wins} wins, restored')
        else:
            reduction = DRAWDOWN_SEVERE_REDUCE
            print(f'  [SIZE] {ticker} — severe drawdown ({drawdown:.1%}), reducing 50%')
    elif drawdown > DRAWDOWN_MILD:
        if consecutive_wins >= CONSECUTIVE_WINS:
            reduction = 0
            print(f'  [SIZE] {ticker} — mild drawdown but {consecutive_wins} wins, restored')
        else:
            reduction = DRAWDOWN_MILD_REDUCE
            print(f'  [SIZE] {ticker} — mild drawdown ({drawdown:.1%}), reducing 30%')
    else:
        reduction = 0

    adjusted_size = base_size * (1 - reduction)

    if high_vol:
        adjusted_size *= 0.50
        print(f'  [SIZE] {ticker} — high realized vol, halving size')

    available_cash = portfolio_value - CASH_FLOOR
    positions_held = sum(1 for s in portfolio_states.values() if s.get('position') != 'flat')
    max_deployable = available_cash / max(positions_held + 1, 1)

    final_size = min(adjusted_size, max_deployable)
    final_size = max(0, final_size)

    print(f'  [SIZE] {ticker} | kelly_f={kelly_f:.3f} | base=£{base_size:.2f} | final=£{final_size:.2f}')
    return round(final_size, 2)


def run_position_sizing(tickers, portfolio_value, regimes, portfolio_states, price_df=None):
    """price_df is optional; when given, checks realised vol per ticker and halves size on a
    high-vol read.
    """
    from layers.regime_detection import is_high_vol_now

    sizes = {}
    for ticker in tickers:
        regime = regimes.get(ticker, 'sideways')
        high_vol = False
        if price_df is not None:
            try:
                high_vol = is_high_vol_now(price_df, ticker)
            except Exception:
                high_vol = False
        sizes[ticker] = compute_position_size(ticker, portfolio_value, regime, portfolio_states, high_vol=high_vol)
    return sizes


if __name__ == '__main__':
    from portfolio_state import initialise_db, get_state

    tickers = ['AAPL', 'MSFT']

    initialise_db()
    portfolio_states = {t: get_state(t) for t in tickers}
    regimes = {'AAPL': 'bull', 'MSFT': 'sideways'}

    sizes = run_position_sizing(tickers, 10000, regimes, portfolio_states)
    print('\nPosition sizes:')
    for t, s in sizes.items():
        print(f'  {t}: £{s}')