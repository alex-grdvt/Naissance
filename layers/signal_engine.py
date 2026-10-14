"""Technical indicators and the long/short alpha scoring functions.

compute_alpha_score and compute_short_score are the core signal layer
consumed by layers.alpha_score (main.py's L3), layers.replacement_logic,
and both backtest engines (backtest/simulation.py, layers/backtest_engine.py).
"""

from datetime import datetime

import pandas as pd

# RSI — canonical implementation lives in layers.trade_logic; re-exported
# here (rather than redefined) so this module's own scoring code, and any
# external caller of signal_engine.compute_rsi, share the one definition.
from layers.trade_logic import compute_rsi, compute_rsi_roc


# MACD
def compute_macd(closes, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram from EMA crossovers."""
    ema_fast = closes.ewm(span=fast).mean()
    ema_slow = closes.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ROC
def compute_roc(closes, period=10):
    """Rate of change over `period` bars, as a percentage."""
    return closes.pct_change(period) * 100


# A/D
def compute_ad_roc(high, low, close, volume, period=5):
    """Rate of change of the A/D line (comparable across start dates,
    unlike the raw cumulative value). Positive = accumulation."""
    mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
    mfv = mfm * volume
    ad = mfv.cumsum()
    return ad.pct_change(period)


# Z-score
def compute_zscore(closes, period=20):
    """Rolling z-score of price against its own `period`-bar mean/std."""
    rolling_mean = closes.rolling(period).mean()
    rolling_std = closes.rolling(period).std()
    return (closes - rolling_mean) / (rolling_std + 1e-8)


# Sentiment
def compute_sentiment_score(sentiment_df, ticker, as_of_date=None):
    """Recency-weighted sentiment score, 0-100 (50 = neutral).

    as_of_date scores sentiment relative to a specific bar date rather
    than wall-clock time, so backtests do not leak future sentiment into
    historical bars. Defaults to today for live use.
    """
    if sentiment_df is None or sentiment_df.empty:
        return 50

    rows = sentiment_df[sentiment_df["ticker"] == ticker]
    if rows.empty:
        return 50

    reference_date = as_of_date if as_of_date is not None else datetime.today()
    reference_date = pd.Timestamp(reference_date)
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_localize(None)

    weighted_sum = 0.0
    total_weight = 0.0

    for _, row in rows.iterrows():
        days_old = (reference_date - pd.to_datetime(row["date"])).days
        if days_old < 0:
            continue  # headline postdates the bar being simulated
        weight = 1 / (days_old + 1)

        if row["label"] == "positive":
            contribution = row["score"] * weight
        elif row["label"] == "negative":
            contribution = -row["score"] * weight
        else:
            contribution = 0

        weighted_sum += contribution
        total_weight += weight

    if total_weight == 0:
        return 50

    avg = weighted_sum / total_weight
    return max(0, min(100, (avg + 1) * 50))


# Long-score regime weights (all sum to 1.0)
WEIGHTS_BULL = {"rsi": 0.25, "macd": 0.30, "roc": 0.20, "ad": 0.15, "sentiment": 0.10}
WEIGHTS_BEAR = {"rsi": 0.25, "macd": 0.25, "roc": 0.20, "ad": 0.20, "sentiment": 0.10}
WEIGHTS_SIDEWAYS = {"rsi": 0.15, "macd": 0.15, "roc": 0.10, "ad": 0.10, "sentiment": 0.10}
ZSCORE_WEIGHT_SIDEWAYS = 0.40  # sideways only; WEIGHTS_SIDEWAYS + this = 1.0
WEIGHTS_HIGH_VOL = {"rsi": 0.25, "macd": 0.20, "roc": 0.20, "ad": 0.20, "sentiment": 0.15}
WEIGHTS_DEFAULT = {"rsi": 0.25, "macd": 0.25, "roc": 0.20, "ad": 0.20, "sentiment": 0.10}


def _rsi_level_score(latest_rsi, regime):
    """RSI level component, 0-100. Bull rewards RSI 50-80 as trend
    confirmation rather than reading it as overbought; other regimes treat
    low RSI as oversold/bullish (mean-reversion read)."""
    if regime == "bull":
        if latest_rsi <= 50:
            return max(0, latest_rsi)
        if latest_rsi <= 80:
            return 50 + (latest_rsi - 50) * (50 / 30)
        return max(20, 100 - (latest_rsi - 80) * 4)
    return max(0, min(100, 100 - latest_rsi))


def compute_alpha_score(price_df, price_df_long, ticker, regime, sentiment_df=None, as_of_date=None):
    """Absolute long score, 0-100. Independent of compute_short_score."""
    closes = price_df["Close"][ticker].dropna()
    high = price_df["High"][ticker].dropna()
    low = price_df["Low"][ticker].dropna()
    volume = price_df["Volume"][ticker].dropna()

    rsi = compute_rsi(closes)
    rsi_roc = compute_rsi_roc(closes)
    _, _, histogram = compute_macd(closes)
    roc = compute_roc(closes)
    ad_roc = compute_ad_roc(high, low, closes, volume)
    zscore = compute_zscore(closes)
    sentiment_score = compute_sentiment_score(sentiment_df, ticker, as_of_date=as_of_date)

    latest_rsi = rsi.iloc[-1]
    latest_rsi_roc = rsi_roc.iloc[-1]
    if pd.isna(latest_rsi_roc):
        latest_rsi_roc = 0.0
    latest_hist = histogram.iloc[-1]
    latest_roc = roc.iloc[-1]
    latest_z = zscore.iloc[-1]
    latest_ad = ad_roc.iloc[-1] if not ad_roc.dropna().empty else 0.0

    current_price = closes.iloc[-1]
    hist_norm = latest_hist / (current_price + 1e-8)

    rsi_level_score = _rsi_level_score(latest_rsi, regime)
    # rising RSI = building momentum, regardless of regime
    rsi_slope_score = max(0, min(100, 50 + latest_rsi_roc * 5))
    # level is the primary read; slope is a secondary confirming nudge
    rsi_score = 0.70 * rsi_level_score + 0.30 * rsi_slope_score

    macd_score = max(0, min(100, 50 + hist_norm * 1000))
    roc_score = max(0, min(100, 50 + latest_roc * 2))
    ad_score = max(0, min(100, 50 + latest_ad * 500))
    z_score = max(0, min(100, 50 - latest_z * 25))

    try:
        from layers.regime_detection import is_high_vol_now
        high_vol = is_high_vol_now(price_df, ticker)
    except Exception:
        high_vol = False

    if high_vol:
        weights = WEIGHTS_HIGH_VOL
        use_zscore = False
    elif regime == "bull":
        weights = WEIGHTS_BULL
        use_zscore = False
    elif regime == "bear":
        weights = WEIGHTS_BEAR
        use_zscore = False
    elif regime == "sideways":
        weights = WEIGHTS_SIDEWAYS
        use_zscore = True  # ZSCORE_WEIGHT_SIDEWAYS applied below
    else:
        weights = WEIGHTS_DEFAULT
        use_zscore = False

    base_score = (
        weights["rsi"] * rsi_score
        + weights["macd"] * macd_score
        + weights["roc"] * roc_score
        + weights["ad"] * ad_score
        + weights["sentiment"] * sentiment_score
    )

    if use_zscore:
        alpha_score = base_score + ZSCORE_WEIGHT_SIDEWAYS * z_score
    else:
        alpha_score = base_score

    if regime == "bear":
        alpha_score = max(0, alpha_score - 15)
    alpha_score = min(100, alpha_score)
    return round(alpha_score, 2)


def _short_rsi_score(rsi_val, overbought=75):
    """RSI component for the short score, centred on 50 at neutral RSI
    (symmetric with the long side's `100 - RSI` baseline)."""
    if rsi_val <= 50:
        return max(0, min(100, rsi_val))
    if rsi_val <= overbought:
        return 50
    return max(50, min(100, 50 + (rsi_val - overbought) * (50 / (100 - overbought))))


def compute_short_score(price_df, ticker, sentiment_df=None, as_of_date=None, regime=None):
    """Absolute short score, 0-100. Independent of compute_alpha_score,
    using different indicators/windows to avoid a long+short≈100 tautology.

    In a bull regime, the overbought bar is raised (RSI 85 vs 75) and
    weight shifts from RSI level onto momentum-decay indicators
    (MACD/ROC20/A-D), since RSI alone cannot distinguish a healthy
    sustained uptrend from genuine blow-off exhaustion.
    """
    closes = price_df["Close"][ticker].dropna()
    high = price_df["High"][ticker].dropna()
    low = price_df["Low"][ticker].dropna()
    volume = price_df["Volume"][ticker].dropna()

    rsi_14 = compute_rsi(closes, period=14)
    rsi_5 = compute_rsi(closes, period=5)  # faster RSI for short-term exhaustion
    rsi_5_roc = compute_rsi_roc(closes, period=5, roc_period=2)
    _, _, histogram = compute_macd(closes)
    roc_20 = compute_roc(closes, period=20)
    ad_roc = compute_ad_roc(high, low, closes, volume, period=10)

    high_52w = closes.rolling(252).max()
    high_52w_clean = high_52w.dropna()
    if not high_52w_clean.empty:
        proximity_high = (high_52w_clean.iloc[-1] - closes.iloc[-1]) / (high_52w_clean.iloc[-1] + 1e-8)
        overbought_bonus = 20 if proximity_high <= 0.05 else 0
    else:
        overbought_bonus = 0

    sentiment_score = compute_sentiment_score(sentiment_df, ticker, as_of_date=as_of_date)

    latest_rsi14 = rsi_14.iloc[-1]
    latest_rsi5 = rsi_5.iloc[-1]
    latest_rsi5_roc = rsi_5_roc.iloc[-1] if not rsi_5_roc.dropna().empty else 0.0
    latest_hist = histogram.iloc[-1]
    latest_roc20 = roc_20.iloc[-1]
    latest_ad = ad_roc.iloc[-1] if not ad_roc.dropna().empty else 0.0

    current_price = closes.iloc[-1]
    hist_norm = latest_hist / (current_price + 1e-8)
    overbought_threshold = 85 if regime == "bull" else 75

    rsi14_score = _short_rsi_score(latest_rsi14, overbought=overbought_threshold)
    rsi5_level_score = _short_rsi_score(latest_rsi5, overbought=overbought_threshold)
    # falling fast-RSI = exhaustion rolling over = bearish
    rsi5_slope_score = max(0, min(100, 50 - latest_rsi5_roc * 5))
    rsi5_score = 0.70 * rsi5_level_score + 0.30 * rsi5_slope_score
    macd_score = max(0, min(100, 50 - hist_norm * 1000))
    roc20_score = max(0, min(100, 50 - latest_roc20 * 2.0))
    ad_score = max(0, min(100, 50 - latest_ad * 500))
    sent_short = max(0, min(100, 100 - sentiment_score))

    if regime == "bull":
        weights = {"rsi14": 0.05, "rsi5": 0.05, "macd": 0.25, "roc20": 0.40, "ad": 0.15, "sent": 0.10}
    else:
        weights = {"rsi14": 0.20, "rsi5": 0.15, "macd": 0.20, "roc20": 0.30, "ad": 0.05, "sent": 0.10}

    short_score = (
        weights["rsi14"] * rsi14_score
        + weights["rsi5"] * rsi5_score
        + weights["macd"] * macd_score
        + weights["roc20"] * roc20_score
        + weights["ad"] * ad_score
        + weights["sent"] * sent_short
    )

    short_score = min(100, short_score + overbought_bonus)
    return round(short_score, 2)


if __name__ == "__main__":
    from dateutil.relativedelta import relativedelta

    from data.price_data import fetch_price_data

    today = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - relativedelta(months=3)).strftime("%Y-%m-%d")
    tickers = ["AAPL", "MSFT"]
    price_df = fetch_price_data(tickers, start, today)

    for ticker in tickers:
        long_s = compute_alpha_score(price_df, price_df, ticker, regime="bull")
        short_s = compute_short_score(price_df, ticker)
        print(f"{ticker}  long={long_s}  short={short_s}  sum={long_s + short_s:.1f}")