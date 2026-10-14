"""
Fetches macro indicators (yield curve spread, VIX, US recession probability,
USD/EUR, USD/CNY, high-yield credit spread) from FRED. Called from main.py to
build macro_df, which feeds layers.kill_switch (L1).
"""

import os
import pandas as pd
from fredapi import Fred
from datetime import datetime


# Recession probability updates lag the underlying economy.
# Data older than this is treated as stale by kill_switch.py.
STALENESS_DAYS = 45


def _get_fred():
    """
    Initialise FRED client.
    The API key is read from the FRED_API_KEY environment variable.
    """
    key = os.environ.get("FRED_API_KEY")

    if not key:
        raise RuntimeError(
            "FRED_API_KEY environment variable is not set."
        )

    return Fred(api_key=key)


def fetch_macro_data(start_date, end_date):
    """Load all macro inputs used by the portfolio-level kill switch."""
    fred = _get_fred()

    # Yield curve inversion signal.
    yield_curve = fred.get_series(
        "T10Y2Y",
        observation_start=start_date,
        observation_end=end_date,
    )

    # Market volatility proxy.
    vix = fred.get_series(
        "VIXCLS",
        observation_start=start_date,
        observation_end=end_date,
    )

    # US recession probability indicator.
    recession = fred.get_series(
        "RECPROUSM156N",
        observation_start=start_date,
        observation_end=end_date,
    )

    # Major FX relationships.
    usd_eur = fred.get_series(
        "DEXUSEU",
        observation_start=start_date,
        observation_end=end_date,
    )

    usd_cny = fred.get_series(
        "DEXCHUS",
        observation_start=start_date,
        observation_end=end_date,
    )

    # High-yield credit stress proxy.
    credit_spread = fred.get_series(
        "BAMLH0A0HYM2",
        observation_start=start_date,
        observation_end=end_date,
    )

    data = pd.DataFrame(
        {
            "yield_curve": yield_curve,
            "vix": vix,
            "recession": recession,
            "usd_eur": usd_eur,
            "usd_cny": usd_cny,
            "credit_spread": credit_spread,
        }
    )

    # Economic series update on different schedules.
    # Forward-fill preserves causality by carrying the most recently available observation forward.
    data = data.ffill()

    # Store the latest recession reading date so downstream logic can detect stale macro inputs.
    data["recession_date"] = recession.last_valid_index()

    return data


if __name__ == "__main__":
    # Simple smoke test.
    today = datetime.today().strftime("%Y-%m-%d")

    df = fetch_macro_data(
        "2010-01-01",
        today,
    )

    print(df.tail())

    print(
        "recession last updated:",
        df["recession_date"].iloc[-1],
    )