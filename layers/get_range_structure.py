"""
Detects support/resistance levels from clustered swing highs and lows, and
flags proximity to or breakout beyond those levels relative to ATR. Used by
stop_loss.py (L8) for range-aware stop placement.
"""

import numpy as np
import pandas as pd

SWING_LOOKBACK = 3  # bars on each side to confirm a swing point

MIN_SUPPORT_TOUCHES = 3  # min swing lows in a cluster to call it support
MIN_RESISTANCE_TOUCHES = 3  # min swing highs in a cluster to call it resistance

LEVEL_CLUSTER_PCT = 0.02  # price tolerance for grouping swings into one level
BREAKOUT_ATR_MULT = 0.50  # ATR multiple beyond the level to call a breakout


def compute_atr(price_df, ticker, period=14):
    """Returns the rolling average true range (ATR) for ticker over period bars."""
    high = price_df['High'][ticker].dropna()
    low = price_df['Low'][ticker].dropna()
    close = price_df['Close'][ticker].dropna()

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def find_swing_points(series, lookback=SWING_LOOKBACK, kind='low'):
    """Returns (index, value) for each local extremum in series confirmed by lookback bars on
    either side.
    """
    vals = series.values
    points = []

    for i in range(lookback, len(vals) - lookback):
        window = vals[i - lookback:i + lookback + 1]
        if kind == 'low' and vals[i] == window.min():
            points.append((i, vals[i]))
        elif kind == 'high' and vals[i] == window.max():
            points.append((i, vals[i]))

    return points


def build_cluster(points, tolerance=LEVEL_CLUSTER_PCT, min_touches=3):
    """
    Groups points within tolerance of each other and returns the mean price
    and touch count of the largest cluster meeting min_touches, or
    (None, touch count) if no cluster qualifies.
    """
    if len(points) < min_touches:
        return None, 0

    prices = sorted(p[1] for p in points)
    best_cluster = []

    for price in prices:
        cluster = [x for x in prices if abs(x - price) / price <= tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_touches:
        return None, len(best_cluster)

    return float(np.mean(best_cluster)), len(best_cluster)


def get_range_structure(price_df, ticker):
    """
    Derives support/resistance levels from clustered swing points, falling
    back to the trailing 60-bar high/low if no cluster qualifies, and flags
    whether price is near either level or has broken out beyond it by
    BREAKOUT_ATR_MULT x ATR.
    """
    closes = price_df['Close'][ticker].dropna()
    lows = price_df['Low'][ticker].dropna()
    highs = price_df['High'][ticker].dropna()

    current_price = closes.iloc[-1]
    atr = compute_atr(price_df, ticker).iloc[-1]

    swing_lows = find_swing_points(lows.iloc[:-1], kind='low')
    swing_highs = find_swing_points(highs.iloc[:-1], kind='high')

    support, support_touches = build_cluster(swing_lows, min_touches=MIN_SUPPORT_TOUCHES)
    resistance, resistance_touches = build_cluster(swing_highs, min_touches=MIN_RESISTANCE_TOUCHES)

    if support is None:
        support = lows.iloc[-60:].min()
    if resistance is None:
        resistance = highs.iloc[-60:].max()

    range_width = resistance - support
    near_support = current_price <= support + 0.20 * range_width
    near_resistance = current_price >= resistance - 0.20 * range_width
    breakout_up = current_price > resistance + BREAKOUT_ATR_MULT * atr
    breakout_down = current_price < support - BREAKOUT_ATR_MULT * atr

    return {
        'support': support,
        'resistance': resistance,
        'support_touches': support_touches,
        'resistance_touches': resistance_touches,
        'range_width': range_width,
        'near_support': near_support,
        'near_resistance': near_resistance,
        'breakout_up': breakout_up,
        'breakout_down': breakout_down,
    }