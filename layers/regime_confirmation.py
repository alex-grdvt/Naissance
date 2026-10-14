"""
Sentiment-based confirmation layer for HMM regime transitions.

Requires a transition to agree with recent news sentiment before it is
accepted; otherwise the previous regime holds for another cycle.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_BULL_MIN = 60
SENTIMENT_BEAR_MAX = 40

MIN_ARTICLES_5D = 3

_REGIME_ORDER = {'bear': 0, 'sideways': 1, 'bull': 2}


def is_upgrade(old_regime: str, new_regime: str) -> bool:
    """True if new_regime ranks above old_regime."""
    return _REGIME_ORDER.get(new_regime, -1) > _REGIME_ORDER.get(old_regime, -1)


def is_downgrade(old_regime: str, new_regime: str) -> bool:
    """True if new_regime ranks below old_regime."""
    return _REGIME_ORDER.get(new_regime, -1) < _REGIME_ORDER.get(old_regime, -1)


def get_recent_sentiment(
    sentiment_df: Optional[pd.DataFrame],
    ticker: str,
    days: int = 5,
    as_of_date=None,
) -> tuple[float, int]:
    """Recency-weighted sentiment score for a ticker (weight = 1/(days_old+1)).

    as_of_date scores sentiment relative to a specific bar date rather than
    wall-clock time, so backtests do not leak future sentiment into historical
    bars (mirrors signal_engine.compute_sentiment_score). Defaults to today
    for live use.

    Returns (score, article_count); score is 0-100, 50 = neutral.
    """
    if sentiment_df is None or sentiment_df.empty:
        return (50.0, 0)

    rows = sentiment_df[sentiment_df['ticker'] == ticker].copy()
    if rows.empty:
        return (50.0, 0)

    if as_of_date is not None:
        reference_date = pd.Timestamp(as_of_date)
    else:
        reference_date = datetime.today()
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_localize(None)

    cutoff = reference_date - timedelta(days=days)

    rows['date'] = pd.to_datetime(rows['date'])
    if rows['date'].dt.tz is not None:
        rows['date'] = rows['date'].dt.tz_convert(None)

    rows = rows[(rows['date'] >= cutoff) & (rows['date'] <= reference_date)]
    if rows.empty:
        return (50.0, 0)

    weighted_sum = 0.0
    total_weight = 0.0
    for _, row in rows.iterrows():
        days_old = (reference_date - row['date']).days
        weight = 1 / (days_old + 1)

        if row['label'] == 'positive':
            contribution = row['score'] * weight
        elif row['label'] == 'negative':
            contribution = -row['score'] * weight
        else:
            contribution = 0.0

        weighted_sum += contribution
        total_weight += weight

    avg = weighted_sum / total_weight if total_weight > 0 else 0.0
    score = max(0.0, min(100.0, (avg + 1) * 50))
    return (round(score, 2), len(rows))


def confirm_regime(
    ticker: str,
    old_regime: str,
    new_regime: str,
    sentiment_df: Optional[pd.DataFrame],
    as_of_date=None,
) -> dict:
    """Confirm an HMM regime transition against recent sentiment.

    Confirmed if enough articles exist and sentiment agrees with the
    move's direction (bullish for upgrade, bearish for downgrade).
    as_of_date threads through to get_recent_sentiment for backtest safety.
    """
    if old_regime == new_regime:
        return {
            'confirmed': True,
            'reason': 'no_change',
            'sentiment_score': None,
            'article_count': None,
            'active_regime': old_regime,
        }

    sentiment_score, article_count = get_recent_sentiment(sentiment_df, ticker, as_of_date=as_of_date)

    if article_count < MIN_ARTICLES_5D:
        logger.info(
            "%s | %s -> %s | PASSTHROUGH (insufficient data: %d articles)",
            ticker, old_regime, new_regime, article_count,
        )
        return {
            'confirmed': True,
            'reason': 'insufficient_data_passthrough',
            'sentiment_score': sentiment_score,
            'article_count': article_count,
            'active_regime': new_regime,
        }

    upgrade = is_upgrade(old_regime, new_regime)
    downgrade = is_downgrade(old_regime, new_regime)

    if upgrade and sentiment_score > SENTIMENT_BULL_MIN:
        confirmed, reason = True, 'confirmed'
    elif downgrade and sentiment_score < SENTIMENT_BEAR_MAX:
        confirmed, reason = True, 'confirmed'
    else:
        confirmed, reason = False, 'sentiment_disagrees'

    active_regime = new_regime if confirmed else old_regime

    logger.info(
        "%s | %s -> %s | %s | sentiment: %.2f | articles: %d",
        ticker, old_regime, new_regime,
        'CONFIRMED' if confirmed else 'UNCONFIRMED',
        sentiment_score, article_count,
    )

    return {
        'confirmed': confirmed,
        'reason': reason,
        'sentiment_score': sentiment_score,
        'article_count': article_count,
        'active_regime': active_regime,
    }


def run_regime_confirmation(
    hmm_regimes: dict[str, str],
    previous_regimes: dict[str, str],
    sentiment_df: Optional[pd.DataFrame],
    as_of_date=None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Apply sentiment confirmation across all tickers for one cycle.

    Returns (confirmed_regimes, confirmation_log) keyed by ticker.
    """
    confirmed_regimes = {}
    confirmation_log = {}
    for ticker, new_regime in hmm_regimes.items():
        old_regime = previous_regimes.get(ticker, new_regime)
        result = confirm_regime(ticker, old_regime, new_regime, sentiment_df, as_of_date=as_of_date)
        confirmed_regimes[ticker] = result['active_regime']
        confirmation_log[ticker] = result
    return confirmed_regimes, confirmation_log
