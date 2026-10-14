"""
Fetches OHLCV price data via yfinance and derives per-ticker daily return and
52-week-high features. Called from main.py to build price_df and
price_df_long, which feed regime_detection (L2), alpha_score (L3),
correlation_monitor (L5) and the sizing/execution layers.
"""

import yfinance as yf
import pandas as pd


def fetch_price_data(tickers, start_date=None, end_date=None, interval="1d", period=None):
    """
    Return OHLCV data and derived price features.

    Yahoo limits intraday history by period rather than date range.
    For non-daily intervals, pass period and ignore start/end dates.
    """

    if isinstance(tickers, str):
        # Treat a single ticker as a one-element universe.
        tickers = [tickers]

    try:

        if period is not None:
            data = yf.download(
                tickers,
                period=period,
                interval=interval,
            )
        else:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
            )

    except Exception as e:
        raise RuntimeError(
            f"yfinance download failed for {tickers}: {e}"
        ) from e

    # Normalise timestamps to timezone-naive US market time.
    if (
        isinstance(data.index, pd.DatetimeIndex)
        and data.index.tz is not None
    ):
        data = data.copy()

        data.index = (
            data.index
            .tz_convert("America/New_York")
            .tz_localize(None)
        )

    for ticker in tickers:

        data[("daily_return", ticker)] = (
            data["Close"][ticker].pct_change()
        )

        # Daily convention uses a 252-bar lookback.
        # Clamp the window when insufficient history exists.
        window = min(252, len(data))

        data[("52w_high", ticker)] = (
            data["Close"][ticker]
            .rolling(window, min_periods=1)
            .max()
        )

    return data


if __name__ == "__main__":

    # Simple smoke test.
    df = fetch_price_data(
        ["AAPL", "MSFT"],
        "2023-01-01",
        "2025-01-01",
    )

    print(df)