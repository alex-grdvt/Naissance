"""backtest/baselines.py — comparison strategies run on the same universe,
trading calendar and starting capital as the main simulation.
"""

import pandas as pd


def _run_buy_and_hold(price_dict, shared_dates, starting_capital):
    """Equal-weight capital across the universe on day 1 and hold to the end, with no trading
    and no costs.
    """
    tickers = list(price_dict.keys())
    alloc_per_ticker = starting_capital / len(tickers)
    shares = {}
    for t in tickers:
        entry_price = price_dict[t]['Close'][t].loc[shared_dates[0]]
        shares[t] = alloc_per_ticker / entry_price

    equity_curve = []
    for date in shared_dates:
        total = sum(shares[t] * price_dict[t]['Close'][t].loc[date] for t in tickers)
        equity_curve.append((date.strftime('%Y-%m-%d'), total))
    return equity_curve


def _run_roc_momentum(price_dict, shared_dates, starting_capital, roc_period=252, rebalance='month'):
    """12-month (252 trading-day) rate-of-change momentum baseline. On the first bar
    of each rebalance period, capital is split equally across tickers whose
    trailing roc_period-bar ROC is positive; the remainder is held in cash.

    No trading costs are applied, unlike the main strategy's ROUND_TRIP_COST_BPS,
    which slightly favours this baseline when comparing Sharpe and CAGR.

    roc_period / rebalance: the defaults (252 bars, monthly) assume daily bars. For
    an intraday run, pass a roc_period scaled to the interval (for example
    BARS_PER_DAY['5m'] bars for one trading day) and rebalance='day'.
    """
    tickers = list(price_dict.keys())
    cash = starting_capital
    shares = {t: 0.0 for t in tickers}
    equity_curve = []
    current_period = None

    for date in shared_dates:
        prices_today = {t: price_dict[t]['Close'][t].loc[date] for t in tickers}
        holdings_value = sum(shares[t] * prices_today[t] for t in tickers)
        total_equity = cash + holdings_value
        equity_curve.append((date.strftime('%Y-%m-%d'), total_equity))

        period_key = (date.year, date.month) if rebalance == 'month' else \
            (date.year, date.month, date.day)
        if period_key != current_period:
            current_period = period_key

            # liquidate everything, then re-derive today's momentum set
            for t in tickers:
                if shares[t] > 0:
                    cash += shares[t] * prices_today[t]
                    shares[t] = 0.0

            in_momentum = []
            for t in tickers:
                closes_upto = price_dict[t]['Close'][t].loc[:date]
                if len(closes_upto) <= roc_period:
                    continue
                roc = closes_upto.iloc[-1] / closes_upto.iloc[-roc_period - 1] - 1
                if roc > 0:
                    in_momentum.append(t)

            if in_momentum:
                alloc = cash / len(in_momentum)
                for t in in_momentum:
                    shares[t] = alloc / prices_today[t]
                    cash -= alloc

    return equity_curve


def _run_spx_benchmark(shared_dates, starting_capital, interval='1d', period=None):
    """Buy-and-hold SPY (S&P 500 ETF proxy) over the same trading calendar and
    starting capital as the main strategy, with no trading and no costs. This is
    the single-ticker version of _run_buy_and_hold, used as the market benchmark.
    """
    from data.price_data import fetch_price_data

    start = shared_dates[0].strftime('%Y-%m-%d')
    end   = shared_dates[-1].strftime('%Y-%m-%d')
    if period is not None:
        spy_df = fetch_price_data(['SPY'], interval=interval, period=period)
    else:
        spy_df = fetch_price_data(['SPY'], start_date=start, end_date=end, interval=interval)

    closes = spy_df['Close']['SPY']
    if isinstance(closes.index, pd.DatetimeIndex) and closes.index.tz is not None:
        closes = closes.copy()
        closes.index = closes.index.tz_localize(None)

    # SPY's trading calendar can differ slightly (holidays, data gaps) from
    # shared_dates, the intersection across the universe. Reindex onto shared_dates
    # and fill forward and back rather than assuming an exact match.
    closes = closes.reindex(shared_dates, method='ffill')
    if closes.isna().any():
        closes = closes.ffill().bfill()

    entry_price = closes.iloc[0]
    shares = starting_capital / entry_price
    equity_curve = [
        (date.strftime('%Y-%m-%d'), float(shares * price))
        for date, price in zip(shared_dates, closes)
    ]
    return equity_curve
