"""
Determines the day's macro_mode (long/neutral/short/cash) from yield curve,
VIX, and recession indicators. Called from main.py as L1, gating the rest of
the pipeline before any per-ticker work runs.
"""

import pandas as pd
from datetime import datetime

STALENESS_DAYS = 45  # recession series lags, warn if older than this


def check_kill_switch(macro_df, as_of_date=None):
    """
    as_of_date scores staleness against the bar being simulated rather than
    wall-clock time, so a historical backtest does not print nonsense "N
    years stale" warnings for every single day. Defaults to today for live use.
    """
    latest = macro_df.iloc[-1]
    reference_date = pd.Timestamp(as_of_date) if as_of_date is not None else datetime.today()
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_localize(None)

    # staleness check on recession series
    recession_date = latest.get('recession_date')
    if recession_date is not None:
        age = (reference_date - pd.Timestamp(recession_date)).days
        if age > STALENESS_DAYS:
            print(f'  [KILL SWITCH] WARNING: recession data is {age} days old — treating as stale')

    yield_inverted = latest['yield_curve'] < 0
    recession_high = latest['recession'] > 20
    vix_extreme = latest['vix'] > 40
    vix_high = latest['vix'] > 30

    if vix_extreme:
        return 'cash'

    if yield_inverted and recession_high and not vix_high:
        return 'short'

    if vix_high or yield_inverted or recession_high:
        return 'neutral'

    return 'long'


if __name__ == '__main__':
    from data.macro_data import fetch_macro_data
    today = datetime.today().strftime('%Y-%m-%d')
    macro_df = fetch_macro_data('2010-01-01', today)
    print(check_kill_switch(macro_df))