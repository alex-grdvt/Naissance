"""Regime-aware trailing/fixed stop-loss logic for long and short positions.

run_stop_loss() is main.py's Layer 8, called after decision_matrix (L6) and
position_sizing (L7); _stop_hit_long/_stop_hit_short in backtest_engine.py
delegate to compute_stop_loss/check_stop_triggered here so backtest and live
stop logic match.
"""

import logging

import numpy as np
import pandas as pd

from layers.get_range_structure import get_range_structure
from layers.regime_detection import is_high_vol_now

logger = logging.getLogger(__name__)

MIN_RANGE_DAYS = 10          # days since entry before trusting a range-based stop
RANGE_BREAK_ATR_MULT = 1.0   # buffer past the established range floor/ceiling
SWING_LOOKBACK = 3           # bars required on both sides to confirm a swing point
MIN_SWING_POINTS = 2         # minimum confirmed swings needed to fit a trendline


def compute_atr(price_df, ticker, period=14):
    """Rolling average true range for ticker over period bars."""
    high = price_df["High"][ticker].dropna()
    low = price_df["Low"][ticker].dropna()
    close = price_df["Close"][ticker].dropna()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def _find_swing_points(series, lookback=SWING_LOOKBACK, kind="low"):
    """Confirmed swing lows/highs only — a bar counts once `lookback` bars
    on both sides are known and it is the extreme of that window, so the
    most recent `lookback` bars can never produce a swing point yet.

    Returns a list of (position, price) tuples, oldest first. Positions
    are integer offsets into `series`, not dates.
    """
    vals = series.values
    n = len(vals)
    pts = []
    for i in range(lookback, n - lookback):
        window = vals[i - lookback: i + lookback + 1]
        if kind == "low" and vals[i] == window.min():
            pts.append((i, vals[i]))
        elif kind == "high" and vals[i] == window.max():
            pts.append((i, vals[i]))
    return pts


def _trendline_value_at(pts, target_idx):
    """OLS line through confirmed swing points, projected to target_idx.

    Returns None if fewer than MIN_SWING_POINTS points are available.
    With exactly 2 widely-spaced or noisy points the projected slope can
    run away — worth validating against real data before relying on the
    2-point minimum.
    """
    if len(pts) < MIN_SWING_POINTS:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope * target_idx + intercept


def compute_stop_loss(price_df, ticker, regime, entry_price, position="long", entry_date=None):
    """Compute a stop price for the given position and regime.

    LONG  — stop sits below price; triggers when price falls too far.
    SHORT — stop sits above price; triggers when price rises too far
            (mirrors backtest_engine's short-side stop logic).

    entry_date bounds the trailing-extreme lookback to the life of this trade
    rather than the whole price history. It is optional, but every caller
    passes it.
    """
    closes = price_df["Close"][ticker].dropna()
    current_price = closes.iloc[-1]

    if isinstance(closes.index, pd.DatetimeIndex) and closes.index.tz is not None:
        closes = closes.copy()
        closes.index = closes.index.tz_convert("America/New_York").tz_localize(None)

    if entry_date is not None:
        entry_date = pd.Timestamp(entry_date)
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_convert("America/New_York").tz_localize(None)

    if entry_price is None:
        return None, "no_position"

    atr = compute_atr(price_df, ticker).iloc[-1]

    # Trailing extremes and the range floor/ceiling are bounded to since-entry
    closes_since_entry = closes.loc[entry_date:] if entry_date is not None else closes

    lows = price_df["Low"][ticker].dropna()
    highs = price_df["High"][ticker].dropna()
    if isinstance(lows.index, pd.DatetimeIndex) and lows.index.tz is not None:
        lows = lows.copy()
        lows.index = lows.index.tz_convert("America/New_York").tz_localize(None)
    if isinstance(highs.index, pd.DatetimeIndex) and highs.index.tz is not None:
        highs = highs.copy()
        highs.index = highs.index.tz_convert("America/New_York").tz_localize(None)
    lows_since_entry = lows.loc[entry_date:] if entry_date is not None else lows
    highs_since_entry = highs.loc[entry_date:] if entry_date is not None else highs

    try:
        high_vol = is_high_vol_now(price_df, ticker)
    except Exception:
        high_vol = False

    if position == "short":
        if high_vol:
            stop = entry_price * 1.05
            return round(stop, 4), "fixed_5pct"

        if regime == "bull":
            return round(current_price, 4), "immediate"  # bull flip invalidates the short thesis

        if regime == "bear":
            rolling_low = closes_since_entry.expanding().min().iloc[-1]
            stop = (rolling_low + 2.5 * atr) if not pd.isna(atr) else (rolling_low * 1.07)
            return round(stop, 4), "trailing_2.5atr"

        if regime == "sideways":
            days_since_entry = len(closes_since_entry) - 1
            if days_since_entry < MIN_RANGE_DAYS or pd.isna(atr):
                stop = (entry_price + 2.0 * atr) if not pd.isna(atr) else (entry_price * 1.05)
                return round(stop, 4), "atr_2.0_from_entry_warmup"

            structure = get_range_structure(price_df, ticker)
            resistance = structure["resistance"]
            stop = resistance + RANGE_BREAK_ATR_MULT * atr
            return round(stop, 4), "sideways_resistance_zone"

        stop = entry_price * 1.05
        return round(stop, 4), "fixed_fallback"

    # LONG (default)
    if high_vol:
        stop = entry_price * 0.95
        return round(stop, 4), "fixed_5pct"

    if regime == "bull":
        rolling_high = closes_since_entry.expanding().max().iloc[-1]
        stop = (rolling_high - 2.5 * atr) if not pd.isna(atr) else (rolling_high * 0.93)
        return round(stop, 4), "trailing_2.5atr"

    if regime == "sideways":
        # Prefers a trendline through confirmed swing lows over a flat
        # floor, so a basing range is not stopped out like a breaking one.
        days_since_entry = len(closes_since_entry) - 1
        if days_since_entry < MIN_RANGE_DAYS or pd.isna(atr):
            stop = (entry_price - 2.0 * atr) if not pd.isna(atr) else (entry_price * 0.95)
            return round(stop, 4), "atr_2.0_from_entry_warmup"

        swing_lows = _find_swing_points(lows_since_entry.iloc[:-1], kind="low")
        target_idx = len(lows_since_entry) - 1
        trend_val = _trendline_value_at(swing_lows, target_idx)

        if trend_val is not None:
            stop = trend_val - RANGE_BREAK_ATR_MULT * atr
            return round(stop, 4), "sideways_support_trendline"

        prior_lows = lows_since_entry.iloc[:-1]
        range_low = prior_lows.min() if len(prior_lows) > 0 else lows_since_entry.iloc[-1]
        stop = range_low - RANGE_BREAK_ATR_MULT * atr
        return round(stop, 4), "sideways_range_floor"

    if regime == "bear":
        return round(current_price, 4), "immediate"

    stop = entry_price * 0.95
    return round(stop, 4), "fixed_fallback"


def check_stop_triggered(price_df, ticker, stop_price, position="long"):
    """LONG triggers at/below the stop; SHORT triggers at/above it."""
    if stop_price is None:
        return False
    current_price = price_df["Close"][ticker].dropna().iloc[-1]
    if position == "short":
        return current_price >= stop_price
    return current_price <= stop_price


def run_stop_loss(price_df, decisions, regimes, portfolio_states):
    """Check every held position's stop and mark triggered ones as EXIT.

    Mutates and returns decisions with stop_triggered/action set on any
    ticker whose stop fired, alongside a stop_actions dict of stop price,
    method, and trigger status per ticker.
    """
    stop_actions = {}

    for ticker, state in portfolio_states.items():
        regime = regimes.get(ticker, "unknown")
        entry_price = state.get("entry_price")
        position = state.get("position", "flat")

        if position == "flat":
            stop_actions[ticker] = {"stop_price": None, "method": "no_position", "triggered": False}
            continue

        entry_date = state.get("entry_date")
        stop_price, method = compute_stop_loss(
            price_df, ticker, regime, entry_price, position=position, entry_date=entry_date
        )
        triggered = check_stop_triggered(price_df, ticker, stop_price, position=position)

        if triggered:
            logger.info("STOP TRIGGERED %s | %s | %s | stop @ %s", ticker, position, method, stop_price)
            if ticker in decisions:
                decisions[ticker]["action"] = "EXIT"
                decisions[ticker]["stop_triggered"] = True
        else:
            logger.debug("stop OK %s | %s | %s | stop @ %s", ticker, position, method, stop_price)

        stop_actions[ticker] = {"stop_price": stop_price, "method": method, "triggered": triggered}

    return stop_actions, decisions


if __name__ == "__main__":
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from data.price_data import fetch_price_data

    today = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - relativedelta(months=6)).strftime("%Y-%m-%d")
    tickers = ["AAPL", "MSFT"]
    price_df = fetch_price_data(tickers, start, today)

    for ticker in tickers:
        current = price_df["Close"][ticker].dropna().iloc[-1]
        print(f"\n{ticker} (current: {current:.2f})")
        for position, entry in [("long", current * 0.95), ("short", current * 1.05)]:
            print(f"  -- {position} --")
            for regime in ["bull", "sideways", "bear"]:
                stop, method = compute_stop_loss(price_df, ticker, regime, entry_price=entry, position=position)
                triggered = check_stop_triggered(price_df, ticker, stop, position=position)
                print(f"  {regime:10s} | {method:25s} | stop: {stop} | triggered: {triggered}")