"""backtest_utils.py — shared backtest utilities: price loading and caching,
bar-frequency helpers, stop-loss delegation, walk-forward Kelly sizing and the
round-trip cost model.

Used by layers/backtest_engine.py (single-ticker backtest) and
run_full_backtest.py (portfolio backtest), so the portfolio backtest does not
depend on backtest_engine.py. Contains no trading decision logic; that lives in
layers/trade_logic.py.
"""

import numpy as np
import pandas as pd

from data.price_data import fetch_price_data
from portfolio_state import cache_prices, get_cached_prices
from layers.position_sizing import compute_kelly, MIN_TRADES as KELLY_MIN_TRADES

BACKTEST_YEARS = 5

# ── Cost model ──
ROUND_TRIP_COST_BPS = 0  # flat round-trip cost (bps); zero for the baseline comparison
SHORT_BORROW_COST_BPS = 0  # extra short-side borrow cost (bps); zero for the baseline comparison


def _round_trip_cost_pct(position):
    """Round-trip cost in percentage points for one trade, direction-aware."""
    bps = ROUND_TRIP_COST_BPS + (SHORT_BORROW_COST_BPS if position == 'short' else 0)
    return bps / 100.0


# ── Position sizing defaults ──
DEFAULT_KELLY_P = 0.50  # fallback win rate for a ticker with no closed-trade history yet
DEFAULT_KELLY_B = 1.20  # fallback win/loss ratio for a ticker with no closed-trade history yet


# ── Bar-frequency helpers ──
# Assumes 1 bar = 1 trading day elsewhere; use BARS_PER_DAY to convert other intervals.
BARS_PER_DAY = {
    '1d': 1, '1wk': 1, '1mo': 1,
    '1h': 7, '60m': 7, '90m': 4,
    '30m': 13, '15m': 26, '5m': 78, '2m': 195, '1m': 390,
}


def periods_per_year(interval):
    """Bars per year for the given interval, for Sharpe annualisation."""
    return BARS_PER_DAY.get(interval, 1) * 252


# ── Price loading ──

def _load_prices(ticker, start, end, interval='1d', period=None):
    """Return an OHLCV DataFrame for ticker, from cache where available, else fetched from
    yfinance.
    """
    if interval != '1d':
        # price_cache keys on (ticker, date) truncated to a calendar day, so cache is bypassed for intraday.
        label = period if period is not None else f"{start} to {end}"
        print(f"  [FETCH] {ticker} — downloading {label} at {interval} from yfinance (cache bypassed for intraday)")
        return fetch_price_data([ticker], start, end, interval=interval, period=period)

    cached = get_cached_prices(ticker, start, end)

    if not cached.empty:
        cached_dates = pd.to_datetime(cached['date'])
        cached_start = cached_dates.min()
        cached_end = cached_dates.max()

        if cached_start <= pd.to_datetime(start) and cached_end >= pd.to_datetime(end):
            print(f"  [CACHE] {ticker} — full coverage from cache")
            return _cache_to_price_df(ticker, cached)

    print(f"  [FETCH] {ticker} — downloading {BACKTEST_YEARS}Y from yfinance")
    price_df = fetch_price_data([ticker], start, end)

    ohlcv = pd.DataFrame({
        'open': price_df['Open'][ticker],
        'high': price_df['High'][ticker],
        'low': price_df['Low'][ticker],
        'close': price_df['Close'][ticker],
        'volume': price_df['Volume'][ticker],
    })
    cache_prices(ticker, ohlcv)

    return price_df


def _cache_to_price_df(ticker, cached_df):
    """Rebuild a yfinance-style multi-index DataFrame from cached OHLCV rows."""
    cached_df = cached_df.copy()
    cached_df['date'] = pd.to_datetime(cached_df['date'])
    cached_df = cached_df.set_index('date')

    df = pd.DataFrame(index=cached_df.index)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[(col.capitalize(), ticker)] = cached_df[col]
    df.columns = pd.MultiIndex.from_tuples(
        [(col.capitalize(), ticker) for col in ['open', 'high', 'low', 'close', 'volume']],
        names=['Price', 'Ticker']
    )
    return df


# ── Stop-loss delegation ──

def _stop_hit_long(price_slice, ticker, regime, entry_price, entry_date=None):
    """True if the long stop loss is breached; delegates to layers.stop_loss so backtest and
    live match.
    """
    from layers.stop_loss import compute_stop_loss, check_stop_triggered
    stop_price, _ = compute_stop_loss(price_slice, ticker, regime, entry_price, position='long', entry_date=entry_date)
    return check_stop_triggered(price_slice, ticker, stop_price, position='long')


def _stop_hit_short(price_slice, ticker, regime, entry_price, entry_date=None):
    """True if the short stop loss is breached; delegates to layers.stop_loss so backtest and
    live match.
    """
    from layers.stop_loss import compute_stop_loss, check_stop_triggered
    stop_price, _ = compute_stop_loss(price_slice, ticker, regime, entry_price, position='short', entry_date=entry_date)
    return check_stop_triggered(price_slice, ticker, stop_price, position='short')


# ── Kelly sizing ──

def _walk_forward_kelly_size(closed_trades, position=None, min_trades=None):
    """Kelly fraction for the next trade, estimated walk-forward from this ticker's own
    closed trades; position restricts the estimate to that direction.
    """
    min_trades = KELLY_MIN_TRADES if min_trades is None else min_trades
    trades = closed_trades if position is None else \
        [t for t in closed_trades if t.get('position') == position]

    if len(trades) < min_trades:
        p, b = DEFAULT_KELLY_P, DEFAULT_KELLY_B
    else:
        pnls = [t['pnl'] for t in trades]
        winners = [x for x in pnls if x > 0]
        losers = [abs(x) for x in pnls if x <= 0]
        p = len(winners) / len(pnls)
        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = np.mean(losers) if losers else 1.0
        b = avg_win / avg_loss if avg_loss > 0 else 0.0

    return compute_kelly(p, b)
