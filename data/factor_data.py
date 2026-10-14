"""
Fetches and locally caches the Kenneth French FF5 + Momentum factor library.
Called from main.py to build factor_df and backtest_factor_df, which feed
alpha_score (L3) and the backtest_engine (L4).
"""

import os
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime


# Local cache for factor datasets.
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    ".cache",
)

FACTORS_CACHE = os.path.join(CACHE_DIR, "ff5_daily.parquet")
MOMENTUM_CACHE = os.path.join(CACHE_DIR, "mom_daily.parquet")

# Refresh factor data monthly.
CACHE_MAX_DAYS = 30


# Kenneth French data library sources.
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


# File formats differ, so parsing offsets must remain separate.
FF5_SKIPROWS = 3
MOM_SKIPROWS = 13


def _cache_fresh(path):
    """Return True if a cached file exists and is recent enough to reuse."""
    if not os.path.exists(path):
        return False

    age_days = (
        datetime.today()
        - datetime.fromtimestamp(os.path.getmtime(path))
    ).days

    return age_days < CACHE_MAX_DAYS


def _download(url, skiprows):
    """Download one zipped Kenneth French factor file and return a DataFrame indexed by
    trading date.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(response.content))

    with zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(
            f,
            skiprows=skiprows,
            index_col=0,
        )

    # Remove footer rows containing annual summaries.
    df = df[
        pd.to_numeric(
            df.index,
            errors="coerce",
        ).notna()
    ]

    df.index = pd.to_datetime(
        df.index,
        format="%Y%m%d",
    )

    return df


def fetch_factor_data(start_date, end_date):
    """Return daily FF5 + Momentum factors for the requested period."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    factors = _load_with_cache(
        FACTORS_CACHE,
        FF5_URL,
        FF5_SKIPROWS,
        "FF5",
    )

    momentum = _load_with_cache(
        MOMENTUM_CACHE,
        MOM_URL,
        MOM_SKIPROWS,
        "Momentum",
    )

    # Common daily factor matrix used by the signal engine.
    combined = factors.join(momentum)

    return combined[start_date:end_date]


def _load_with_cache(cache_path, url, skiprows, label):
    """Read from cache when possible. Re-download if the cache is missing or stale."""
    if _cache_fresh(cache_path):
        return pd.read_parquet(cache_path)

    print(f"[FACTORS] downloading {label}...")

    try:
        df = _download(url, skiprows)
        df.to_parquet(cache_path)

        return df

    except Exception as e:

        # Fall back to stale data if a download fails.
        if os.path.exists(cache_path):
            print(
                f"[FACTORS] {label} download failed ({e}), "
                f"using cached copy"
            )

            return pd.read_parquet(cache_path)

        raise


if __name__ == "__main__":
    # Simple smoke test.
    today = datetime.today().strftime("%Y-%m-%d")

    df = fetch_factor_data(
        "2010-01-01",
        today,
    )

    print(df.tail())
    print("shape:", df.shape)