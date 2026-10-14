"""
Computes calibrated long-side alpha scores per ticker via signal_engine and
factor_decomposition, applies macro-mode scaling, and attaches the
independently computed short-side score for downstream position sizing.
Called from main.py's L3 (score_all_tickers).
"""

import pandas as pd
from layers.signal_engine import compute_alpha_score, compute_short_score
from layers.factor_decomposition import apply_factor_penalty

# Valid macro_mode values are 'long', 'neutral', 'short', and 'cash'; 'cash' is kept
# here as a safety default even though score_all_tickers already short-circuits on it.
MODE_MULTIPLIERS = {
    'long': 1.0,
    'neutral': 0.5,
    'short': 0.2,
    'cash': 0.0,
}


def compute_final_score(price_df, price_df_long, factor_df, sentiment_df, ticker, regime, macro_mode):
    """Compute the long-side score for one ticker under the given regime and macro_mode."""
    raw_score = compute_alpha_score(price_df, price_df_long, ticker, regime, sentiment_df)

    # betas is not used here; factor exposure is captured by alpha and r_squared alone.
    factor_adjusted_score, alpha, _betas, r_squared = apply_factor_penalty(
        raw_score, price_df, factor_df, ticker
    )

    multiplier = MODE_MULTIPLIERS.get(macro_mode, 1.0)
    final_score = factor_adjusted_score * multiplier

    return round(final_score, 2), {
        'raw_score': raw_score,
        'factor_adjusted_score': round(factor_adjusted_score, 2),  # pre-macro-multiplier;
        # this is what backtest_engine.py / run_full_backtest.py call
        # long_score, and what trade_logic's gates were calibrated against.
        # final_score below is macro-mode-scaled and must NOT be used for
        # entry-gate comparisons (decision_matrix.py uses this field, not
        # final_score, for exactly that reason).
        'factor_alpha': alpha,
        'r_squared': r_squared,
        'macro_mode': macro_mode,
        'regime': regime,
        'final_score': round(final_score, 2),
    }


def score_all_tickers(price_df, price_df_long, factor_df, sentiment_df, regimes, macro_mode):
    """Compute long and short scores for every ticker in regimes.

    Short-circuits to an empty dict under macro_mode 'cash'. Per-ticker
    failures are caught and logged individually so one ticker's exception
    does not abort scoring for the rest of the batch.
    """
    if macro_mode == 'cash':
        print('kill switch: cash mode — no scores computed')
        return {}

    scores = {}
    for ticker, regime in regimes.items():
        try:
            final_score, breakdown = compute_final_score(
                price_df, price_df_long, factor_df, sentiment_df, ticker, regime, macro_mode
            )

            # short_score is computed regardless of macro_mode and stored for
            # downstream position sizing; it is not blended into final_score here.
            short_score = compute_short_score(price_df, ticker, sentiment_df=sentiment_df, regime=regime)
            breakdown['short_score'] = short_score

            scores[ticker] = breakdown

            alpha_str = f'{breakdown["factor_alpha"]:.6f}' if breakdown['factor_alpha'] else 'n/a'
            print(f'\n{ticker}')
            print(f' regime: {regime}')
            print(f' raw score (long): {breakdown["raw_score"]}')
            print(f' short score: {short_score}')
            print(f' factor alpha: {alpha_str}')
            print(f' final score (long, mode-adjusted): {final_score}')

        except Exception:
            import traceback
            traceback.print_exc()

    return scores


if __name__ == '__main__':
    from data.price_data import fetch_price_data
    from data.factor_data import fetch_factor_data
    from data.macro_data import fetch_macro_data
    from data.sentiment_data import fetch_sentiment_data
    from layers.kill_switch import check_kill_switch
    from layers.regime_detection import detect_regime
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    today = datetime.today().strftime('%Y-%m-%d')
    signal_start = (datetime.today() - relativedelta(months=13)).strftime('%Y-%m-%d')
    regime_start = '2020-01-01'
    tickers = ['AAPL', 'MSFT']

    price_df_long = fetch_price_data(tickers, regime_start, today)
    price_df = fetch_price_data(tickers, signal_start, today)
    factor_df = fetch_factor_data(regime_start, today)
    macro_df = fetch_macro_data(signal_start, today)
    sentiment_df = fetch_sentiment_data(tickers, signal_start, today)

    macro_mode = check_kill_switch(macro_df)
    print('macro mode:', macro_mode)

    regimes = {}
    for ticker in tickers:
        _, current_regime, _ = detect_regime(price_df_long, ticker)
        regimes[ticker] = current_regime

    scores = score_all_tickers(price_df, price_df_long, factor_df, sentiment_df, regimes, macro_mode)