"""backtest_engine.py — single-ticker, day-by-day backtest runner.

A helper layer rather than a second decision engine: price loading, indicators,
stop-loss delegation, walk-forward Kelly sizing, cost and summary statistics,
and the database-facing run_backtest() entry point. The entry and exit decision
is owned by layers.trade_logic, which _simulate() calls.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from data.price_data import fetch_price_data
from portfolio_state import (
    get_connection,
    cache_prices,
    get_cached_prices,
    log_trade,
    log_backtest_results,
    update_maturity,
    get_maturity,
)
from layers.position_sizing import compute_kelly, MIN_TRADES as KELLY_MIN_TRADES

# Gate constants and entry/exit logic live in layers.trade_logic. They are
# re-exported here so that `from layers.backtest_engine import SCORE_GAP` keeps
# working.
from layers.trade_logic import (
    SCORE_GAP,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    FAST_MOM_THRESH,
    FAST_MOM_THRESH_SHORT,
    compute_rsi as _compute_rsi,
    compute_rsi_roc as _compute_rsi_roc,
    long_entry_gates,
    short_entry_gates,
    long_exit_due_to_regime,
    short_exit_due_to_regime,
)

BACKTEST_YEARS = 5
MIN_TRADES_SKIP = 10  # skip ticker if this many backtest trades already exist

# ── Cost model ──
ROUND_TRIP_COST_BPS = 0  # flat round-trip cost (bps); zero for the baseline comparison
SHORT_BORROW_COST_BPS = 0  # extra short-side borrow cost (bps); zero for the baseline comparison


def _round_trip_cost_pct(position):
    """Round-trip cost in percentage points for one trade, direction-aware."""
    bps = ROUND_TRIP_COST_BPS + (SHORT_BORROW_COST_BPS if position == 'short' else 0)
    return bps / 100.0


# ── Position sizing ──
DEFAULT_KELLY_P = 0.50  # fallback win rate for a ticker with no closed-trade history yet
DEFAULT_KELLY_B = 1.20  # fallback win/loss ratio for a ticker with no closed-trade history yet


# ── Helpers ──

def _count_backtest_trades(ticker):
    """Number of existing backtest-sourced rows for ticker in trade_log."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE ticker = ? AND source = 'backtest'",
            (ticker,)
        ).fetchone()
    return row[0]


# ── Bar-frequency helpers ──
# Assumes 1 bar = 1 trading day elsewhere in this file; use BARS_PER_DAY to convert other intervals.
BARS_PER_DAY = {
    '1d': 1, '1wk': 1, '1mo': 1,
    '1h': 7, '60m': 7, '90m': 4,
    '30m': 13, '15m': 26, '5m': 78, '2m': 195, '1m': 390,
}


def periods_per_year(interval):
    """Bars per year for the given interval, for Sharpe annualisation."""
    return BARS_PER_DAY.get(interval, 1) * 252


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


def _compute_atr(price_slice, ticker, period=14):
    """Rolling average true range for ticker over period bars."""
    high = price_slice['High'][ticker]
    low = price_slice['Low'][ticker]
    close = price_slice['Close'][ticker]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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


def _record_exit(trades, ticker, position_type, pos, exit_price, exit_date_str, regime, reason):
    """Append a completed trade dict to trades and print an EXIT log line."""
    if position_type == 'long':
        pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
    else:
        pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * 100

    trades.append({
        'position': position_type,
        'entry_price': pos['entry_price'],
        'exit_price': exit_price,
        'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
        'exit_date': exit_date_str,
        'pnl': pnl,
        'regime': regime,
        'size': pos['size'],
    })
    print(f"EXIT {ticker} | reason={reason} | regime={regime}")


# ── Core simulation ──

def _simulate(price_df, ticker, factor_df, sentiment_df=None, enable_shorts=False):
    """Walk the price series bar by bar, scoring and applying entry/exit logic; returns
    completed trades.
    """
    from layers.regime_detection import detect_regime, detect_regime_walkforward
    from layers.signal_engine import compute_alpha_score, compute_short_score
    from layers.factor_decomposition import apply_factor_penalty_both

    closes = price_df['Close'][ticker].dropna()
    dates = closes.index

    MIN_WARMUP = 300

    # detect_regime_walkforward avoids the look-ahead leak of fitting the HMM once on the full series.
    USE_WALKFORWARD_REGIME = True
    try:
        if USE_WALKFORWARD_REGIME:
            regime_series = detect_regime_walkforward(price_df, ticker)
        else:
            regime_series, _, _ = detect_regime(price_df, ticker)
    except Exception as e:
        print(f"  [BACKTEST] {ticker} — regime detection failed: {e}")
        return []

    full_daily_ret = closes.pct_change()

    long_pos = None
    short_pos = None
    trades = []

    for i in range(MIN_WARMUP, len(dates)):
        bar_date = dates[i]
        price_slice = {col: price_df[col][ticker].iloc[:i + 1]
                       for col in price_df.columns.get_level_values(0).unique()}

        slice_df = pd.concat(price_slice, axis=1)
        slice_df.columns = pd.MultiIndex.from_tuples(
            [(c, ticker) for c in price_slice.keys()],
            names=['Price', 'Ticker']
        )
        slice_df[('daily_return', ticker)] = full_daily_ret.iloc[:i + 1].values

        current_price = closes.iloc[i]
        bar_date_str = bar_date.strftime('%Y-%m-%d')

        if bar_date not in regime_series.index:
            continue
        regime = regime_series.loc[bar_date]

        try:
            raw_long = compute_alpha_score(
                slice_df, slice_df, ticker, regime, sentiment_df=sentiment_df, as_of_date=bar_date
            )
            raw_short = compute_short_score(slice_df, ticker, sentiment_df=sentiment_df, as_of_date=bar_date, regime=regime)
            try:
                long_score, short_score, _, _, _ = apply_factor_penalty_both(
                    raw_long, raw_short, slice_df, factor_df, ticker, end_date=bar_date
                )
            except Exception:
                long_score, short_score = raw_long, raw_short
        except Exception:
            continue

        rsi_series = _compute_rsi(closes.iloc[:i + 1])
        rsi = rsi_series.iloc[-1] if not rsi_series.dropna().empty else 50.0

        rsi_roc_series = _compute_rsi_roc(closes.iloc[:i + 1])
        rsi_roc = rsi_roc_series.iloc[-1] if not rsi_roc_series.dropna().empty else 0.0

        if long_pos is not None:
            stop_hit = _stop_hit_long(slice_df, ticker, regime, long_pos['entry_price'], entry_date=pd.to_datetime(long_pos['entry_date']))
            if stop_hit or long_exit_due_to_regime(regime):
                _record_exit(trades, ticker, 'long', long_pos, current_price, bar_date_str, regime,
                             reason='STOP' if stop_hit else 'REGIME')
                long_pos = None

        if short_pos is not None:
            stop_hit = _stop_hit_short(slice_df, ticker, regime, short_pos['entry_price'], entry_date=pd.to_datetime(short_pos['entry_date']))
            if stop_hit or short_exit_due_to_regime(regime):
                _record_exit(trades, ticker, 'short', short_pos, current_price, bar_date_str, regime,
                             reason='STOP' if stop_hit else 'REGIME')
                short_pos = None

        if long_pos is None and short_pos is None:
            wants_long, _ = long_entry_gates(long_score, short_score, rsi, rsi_roc, regime)
            wants_short, _ = short_entry_gates(long_score, short_score, rsi, rsi_roc, regime,
                                                enable_shorts=enable_shorts)
            # long_score > short_score + GAP and short_score > long_score + GAP cannot both
            # hold for GAP > 0, so wants_long/wants_short are never both true.
            if wants_long:
                long_pos = {
                    'entry_price': current_price,
                    'entry_date': bar_date,  # real Timestamp, needed for tz-aware stop comparisons
                    'regime': regime,
                    'size': _walk_forward_kelly_size(trades, position='long'),
                }
            elif wants_short:
                short_pos = {
                    'entry_price': current_price,
                    'entry_date': bar_date,
                    'regime': regime,
                    'size': _walk_forward_kelly_size(trades, position='short'),
                }

    if long_pos is not None:
        _record_exit(trades, ticker, 'long', long_pos, closes.iloc[-1], dates[-1].strftime('%Y-%m-%d'),
                     long_pos['regime'], reason='END_OF_DATA')

    if short_pos is not None:
        _record_exit(trades, ticker, 'short', short_pos, closes.iloc[-1], dates[-1].strftime('%Y-%m-%d'),
                     short_pos['regime'], reason='END_OF_DATA')

    return trades


# ── Summary stats ──

def _compute_summary(trades):
    """Compute trade summary stats from a compounding, cost- and Kelly-size-adjusted equity
    curve.
    """
    if not trades:
        return None

    gross_pnls = [t['pnl'] for t in trades]
    net_pnls = [p - _round_trip_cost_pct(t.get('position', 'long')) for p, t in zip(gross_pnls, trades)]

    default_size = compute_kelly(DEFAULT_KELLY_P, DEFAULT_KELLY_B)
    sizes = [t.get('size', default_size) for t in trades]

    winners = [p for p in net_pnls if p > 0]
    losers = [p for p in net_pnls if p <= 0]

    n = len(net_pnls)
    win_rate = len(winners) / n
    avg_win = np.mean(winners) if winners else 0.0
    avg_loss = abs(np.mean(losers)) if losers else 0.0

    weighted_returns = [(p / 100.0) * s for p, s in zip(net_pnls, sizes)]  # trades are sequential, so no overlap risk

    equity = 1.0
    equity_curve = [equity]
    for r in weighted_returns:
        equity *= (1 + r)
        equity_curve.append(equity)
    equity_curve = np.array(equity_curve)

    arr = np.array(weighted_returns)
    sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    max_dd = float(drawdowns.max()) * 100

    total_return_pct = (float(equity_curve[-1]) - 1) * 100
    first_date = pd.to_datetime(trades[0]['entry_date'])
    last_date = pd.to_datetime(trades[-1]['exit_date'])
    years_elapsed = max((last_date - first_date).days / 365.25, 1 / 365.25)  # actual trade span, not a fixed BACKTEST_YEARS
    annualized_return_pct = (float(equity_curve[-1]) ** (1 / years_elapsed) - 1) * 100

    avg_cost_drag_pct = float(np.mean([g - net for g, net in zip(gross_pnls, net_pnls)]))

    return {
        'n_trades': n,
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
        'final_equity': round(float(equity_curve[-1]), 4),
        'total_return_pct': round(total_return_pct, 4),
        'annualized_return_pct': round(annualized_return_pct, 4),
        'cost_bps_per_trade': ROUND_TRIP_COST_BPS,
        'avg_cost_drag_pct': round(avg_cost_drag_pct, 4),
        'avg_position_size': round(float(np.mean(sizes)), 4),
    }


def _resolve_maturity(ticker, n_live_trades, n_backtest_trades):
    """Combine live and backtest trade counts into a maturity tier.

    Returns (total, sentiment_active, kelly_mode): sentiment_active gates
    real sentiment use, kelly_mode selects which win-rate/b source
    position_sizing uses ('default', 'backtest', or 'live').
    """
    total = n_live_trades + n_backtest_trades
    sentiment_active = total >= 50
    if total < 10:
        kelly_mode = 'default'
    elif total < 50:
        kelly_mode = 'backtest'
    else:
        kelly_mode = 'live'
    return total, sentiment_active, kelly_mode


# ── Public entry point ──

def run_backtest(ticker, factor_df, sentiment_df=None):
    """Run the backtest for ticker unless MIN_TRADES_SKIP backtest rows already exist;
    returns the summary dict or None.
    """
    existing = _count_backtest_trades(ticker)
    if existing >= MIN_TRADES_SKIP:
        print(f"  [BACKTEST] {ticker} — {existing} rows already in DB, skipping")
        return None

    today = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - relativedelta(years=BACKTEST_YEARS)).strftime('%Y-%m-%d')

    price_df = _load_prices(ticker, start, today)
    if price_df is None or price_df.empty:
        print(f"  [BACKTEST] {ticker} — price fetch failed, aborting")
        return None

    if sentiment_df is None:
        try:
            from data.sentiment_data import fetch_sentiment_data
            sentiment_df = fetch_sentiment_data([ticker], start, today)
        except Exception as e:
            print(f"  [BACKTEST] {ticker} — sentiment fetch failed, proceeding neutral: {e}")
            sentiment_df = None

    print(f"  [BACKTEST] {ticker} — simulating {BACKTEST_YEARS}Y …")
    trades = _simulate(price_df, ticker, factor_df, sentiment_df=sentiment_df)

    if not trades:
        print(f"  [BACKTEST] {ticker} — no trades generated")
        return None

    for t in trades:
        log_trade(
            ticker=ticker,
            action='EXIT',
            position=t['position'],
            entry_price=t['entry_price'],
            exit_price=t['exit_price'],
            entry_date=t['entry_date'],
            regime=t['regime'],
            alpha_score=None,
            source='backtest',
        )

    summary = _compute_summary(trades)
    log_backtest_results(
        ticker=ticker,
        n_trades=summary['n_trades'],
        win_rate=summary['win_rate'],
        avg_win=summary['avg_win'],
        avg_loss=summary['avg_loss'],
        sharpe=summary['sharpe'],
        max_drawdown=summary['max_drawdown'],
        total_return_pct=summary['total_return_pct'],
        annualized_return_pct=summary['annualized_return_pct'],
    )

    with get_connection() as conn:
        live_trades = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE ticker = ? AND source = 'live'", (ticker,)
        ).fetchone()[0]

    total, sentiment_active, kelly_mode = _resolve_maturity(ticker, live_trades, summary['n_trades'])
    update_maturity(ticker, total, sentiment_active, kelly_mode)

    print(
        f"  [BACKTEST] {ticker} — done | "
        f"n={summary['n_trades']} | win={summary['win_rate']:.1%} | "
        f"sharpe={summary['sharpe']:.2f} | kelly_mode={kelly_mode}"
    )

    return summary


if __name__ == '__main__':
    from data.factor_data import fetch_factor_data
    from portfolio_state import initialise_db

    initialise_db()

    today = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d')
    factor_df = fetch_factor_data(start, today)

    result = run_backtest('AAPL', factor_df)
    if result:
        print("\nSummary:", result)