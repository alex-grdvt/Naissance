"""
Flags overly correlated pairs and portfolio-wide correlation risk among
current holdings, using trailing daily returns. Called from main.py's L5
(correlation_monitor).
"""

import pandas as pd
import numpy as np

PAIR_THRESHOLD = 0.7  # flag if any two stocks move together this much
AVERAGE_THRESHOLD = 0.6  # flag if the whole portfolio is too correlated
LOOKBACK_DAYS = 30  # rolling window


def compute_correlation_matrix(price_df, holdings):
    """Returns the pairwise correlation matrix of daily returns for holdings over the
    trailing LOOKBACK_DAYS.
    """
    returns = price_df['daily_return'][holdings].dropna().tail(LOOKBACK_DAYS)
    return returns.corr()


def flag_high_correlations(corr_matrix):
    """Returns (ticker_a, ticker_b, correlation) for every pair exceeding PAIR_THRESHOLD."""
    flagged_pairs = []
    tickers = corr_matrix.columns

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):  # upper triangle only, avoid duplicate pairs
            corr = corr_matrix.iloc[i, j]
            if corr > PAIR_THRESHOLD:
                flagged_pairs.append((tickers[i], tickers[j], round(corr, 3)))

    return flagged_pairs


def get_average_correlation(corr_matrix):
    """Returns the mean off-diagonal correlation in corr_matrix."""
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg = corr_matrix.values[mask].mean()
    return round(avg, 3)


def apply_direction_signs(corr_matrix, directions):
    """
    Multiply each pairwise correlation by the product of direction signs
    (+1 long, -1 short) so downstream flagging uses risk-relevant, not raw
    price, correlation: same direction stacks risk (flag), opposite
    direction hedges (do not flag). directions is {ticker: 'long'|'short'}
    for every ticker in corr_matrix. All-long input is a no-op.
    """
    signs = pd.Series({t: (1.0 if directions.get(t, 'long') == 'long' else -1.0)
                        for t in corr_matrix.columns})
    sign_matrix = pd.DataFrame(
        np.outer(signs.values, signs.values),
        index=corr_matrix.index, columns=corr_matrix.columns
    )
    return corr_matrix * sign_matrix


def run_correlation_monitor(price_df, holdings):
    """
    Computes the holdings' correlation matrix, flags high pairwise
    correlations, and flags the whole portfolio if average correlation
    exceeds AVERAGE_THRESHOLD. Requires at least two holdings; returns empty
    results and a false flag otherwise.
    """
    if len(holdings) < 2:
        print("need at least 2 holdings to compute correlations")
        return pd.DataFrame(), [], False

    corr_matrix = compute_correlation_matrix(price_df, holdings)
    flagged_pairs = flag_high_correlations(corr_matrix)
    avg_corr = get_average_correlation(corr_matrix)
    portfolio_flag = avg_corr > AVERAGE_THRESHOLD

    print("\ncorrelation monitor")
    print(f" average portfolio correlation: {avg_corr}")
    if portfolio_flag:
        print(f" WARNING: portfolio too correlated (>{AVERAGE_THRESHOLD})")

    if flagged_pairs:
        print(f" flagged pairs (>{PAIR_THRESHOLD}):")
        for a, b, corr in flagged_pairs:
            print(f"   {a} / {b}: {corr}")
    else:
        print(" no highly correlated pairs")

    return corr_matrix, flagged_pairs, portfolio_flag


if __name__ == '__main__':
    from data.price_data import fetch_price_data
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    today = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - relativedelta(months=3)).strftime('%Y-%m-%d')
    holdings = ['TSLA', 'NVDA']

    price_df = fetch_price_data(holdings, start, today)
    run_correlation_monitor(price_df, holdings)