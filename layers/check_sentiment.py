"""Diagnostic: compute_sentiment_score output across synthetic sentiment data modes."""

import sys
sys.path.insert(0, '.')

from test_synthetic import build_synthetic_prices, build_synthetic_sentiment, TICKER
from signal_engine import compute_sentiment_score

price_df, phase_series, kind_series = build_synthetic_prices()

for mode in ['none', 'noisy']:
    sentiment_df = build_synthetic_sentiment(phase_series, kind_series, mode=mode)
    score = compute_sentiment_score(sentiment_df, TICKER)
    n_rows = 0 if sentiment_df is None else len(sentiment_df)
    print(f'mode={mode:8s} rows={n_rows:4d} sentiment_score={score}')