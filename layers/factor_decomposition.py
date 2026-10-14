"""
Runs FF5 + momentum factor regressions per ticker and converts the resulting
alpha into a penalty multiplier applied to raw alpha scores, discounting
scores that show no edge beyond factor exposure. Used by alpha_score.py (L3)
and, for the backtest loop, by backtest_engine.py/backtest_utils.py.
"""

import pandas as pd
import statsmodels.api as sm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimum daily alpha considered a real edge (0.02%).
ALPHA_THRESHOLD = 0.0002


def run_factor_regression(price_df, factor_df, ticker, end_date=None):
    """Run FF5 + momentum OLS regression for ticker. end_date truncates factor_df to avoid
    look-ahead leakage.
    """
    returns = price_df['daily_return'][ticker].dropna()

    # Strip tz so returns can join the tz-naive factor frame.
    if isinstance(returns.index, pd.DatetimeIndex) and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_convert('America/New_York').tz_localize(None)

    # Factors are daily; compound intraday bars to daily or the join is empty and OLS fails every bar.
    if isinstance(returns.index, pd.DatetimeIndex) and \
            ((returns.index.hour != 0).any() or returns.index.has_duplicates):
        returns = returns.resample('1D').apply(lambda x: (1 + x).prod() - 1).dropna()

    if end_date is not None:
        # end_date can be tz-aware from the intraday loop; strip tz or the compare raises.
        end_date = pd.Timestamp(end_date)
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        f = factor_df.loc[:end_date]
    else:
        f = factor_df

    combined = returns.to_frame().join(f, how='inner')

    # Fail loudly on a near-empty overlap instead of letting the OLS choke and the caller swallow it silently.
    if len(combined) < 60:
        raise ValueError(
            f'only {len(combined)} overlapping daily obs '
            f'(returns end {returns.index[-1].date()}, '
            f'factors end {f.index[-1].date() if len(f) else "EMPTY"})'
        )

    combined.columns = ['daily_return', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
    combined['excess_return'] = combined['daily_return'] - combined['RF'] / 100

    y = combined['excess_return']
    X = combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']]
    X = sm.add_constant(X, has_constant='add')

    model = sm.OLS(y, X).fit()

    alpha = model.params['const']
    betas = model.params.drop('const')
    r_squared = model.rsquared

    return alpha, betas, r_squared


def _factor_penalty_multiplier(alpha, direction='long'):
    """Penalty multiplier for alpha in the given direction. See apply_factor_penalty for the
    sign rationale.
    """
    if direction == 'short':
        if alpha > 0:
            return 0.5  # real edge argues against shorting
        elif abs(alpha) < ALPHA_THRESHOLD:
            return 0.7  # negligible alpha
        else:
            return 1.0  # negative alpha confirms the short
    else:
        if alpha < 0:
            return 0.5  # penalise hard
        elif abs(alpha) < ALPHA_THRESHOLD:
            return 0.7  # negligible alpha
        else:
            return 1.0  # genuine edge


def apply_factor_penalty(alpha_score, price_df, factor_df, ticker, end_date=None, direction='long'):
    """
    Adjust alpha_score down if the stock has no real edge beyond factor exposure.

    direction mirrors the penalty logic for shorts, not just the sign:
        long:  negative alpha penalises, positive alpha beyond threshold gets full credit.
        short: negative alpha beyond threshold confirms the thesis (full credit), positive alpha penalises hard.
        both:  |alpha| under threshold is a 30% cut either way.

    Use apply_factor_penalty_both for the per-bar backtest loops; it runs the regression once
    instead of twice.
    """
    try:
        alpha, betas, r_squared = run_factor_regression(
            price_df, factor_df, ticker, end_date=end_date
        )
        penalty = _factor_penalty_multiplier(alpha, direction)
        adjusted_score = round(alpha_score * penalty, 2)
        return adjusted_score, alpha, betas, r_squared

    except Exception as e:
        print(f'  factor regression failed for {ticker}: {e}')
        return round(alpha_score, 2), None, None, None


def apply_factor_penalty_both(long_score, short_score, price_df, factor_df, ticker, end_date=None):
    """
    Run the regression once and apply both long and short penalties to it.

    Avoids doubling the per-bar regression cost when a caller needs both directions.
    Returns (adjusted_long, adjusted_short, alpha, betas, r_squared); on failure, returns
    the original scores unchanged.
    """
    try:
        alpha, betas, r_squared = run_factor_regression(
            price_df, factor_df, ticker, end_date=end_date
        )
        adjusted_long = round(long_score * _factor_penalty_multiplier(alpha, 'long'), 2)
        adjusted_short = round(short_score * _factor_penalty_multiplier(alpha, 'short'), 2)
        return adjusted_long, adjusted_short, alpha, betas, r_squared

    except Exception as e:
        print(f'  factor regression failed for {ticker}: {e}')
        return round(long_score, 2), round(short_score, 2), None, None, None


if __name__ == '__main__':
    from data.price_data import fetch_price_data
    from data.factor_data import fetch_factor_data
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    today = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - relativedelta(months=3)).strftime('%Y-%m-%d')

    tickers = ['AAPL', 'MSFT']
    price_df = fetch_price_data(tickers, '2020-01-01', today)
    factor_df = fetch_factor_data('2020-01-01', today)

    for ticker in tickers:
        alpha, betas, r2 = run_factor_regression(price_df, factor_df, ticker)
        print(f'\n{ticker} alpha={alpha:.6f} r2={r2:.4f}')