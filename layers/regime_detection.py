"""HMM-based bull/bear/sideways regime detection, plus an independent
realised-volatility signal.

detect_regime is main.py's L2 regime detector; detect_regime_walkforward
and compute_high_vol_series/is_high_vol_now are used throughout the
backtest engines (backtest/simulation.py, layers/backtest_engine.py) and
sizing/stop-loss layers.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# Multi-restart HMM fit: EM only finds a local optimum, so the model is fitted
# N_RESTARTS times on standardised features and the best converged run is kept.
N_RESTARTS = 8
BASE_SEED = 42
N_ITER = 1000
TOL = 1e-4
MIN_COVAR = 1e-3  # floor on diagonal covariance to prevent state collapse

# A state's real (unstandardised) mean 20d return must clear this to be
# labelled bear/bull; otherwise it is sideways. A ticker can lack a bear or
# bull state entirely if nothing in its history qualifies.
BEAR_RET20_THRESHOLD = -0.02
BULL_RET20_THRESHOLD = 0.02
BEAR_ENABLED = True  # kept as a kill switch for bear classification

HMM_MIN_BARS = 200  # minimum feature rows required to fit


def _fit_one(features_scaled, n_states, seed):
    """Fit one GaussianHMM at the given seed.

    Returns (model, converged, loglik), or None if the fit failed or
    collapsed (NaN start/transition probabilities).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=N_ITER,
            tol=TOL,
            min_covar=MIN_COVAR,
            random_state=seed,
        )
        try:
            model.fit(features_scaled)
        except Exception:
            return None

        if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
            return None

        converged = model.monitor_.converged and not any(
            "is not converging" in str(w.message) for w in caught
        )
        try:
            loglik = model.score(features_scaled)
        except Exception:
            return None

    return model, converged, loglik


def label_states(model, feat_mean=None, feat_std=None):
    """Map HMM state indices to bear/bull/sideways labels.

    States are ranked on their real (unstandardised) mean 20d return
    against BEAR_RET20_THRESHOLD / BULL_RET20_THRESHOLD. Volatility is not
    used here — see compute_high_vol_series() for that as a separate,
    independent signal consumed directly by stop_loss/position_sizing.

    feat_mean/feat_std unstandardise model.means_ back to real return
    units; if omitted, falls back to ranking on standardised means.
    """
    means = pd.DataFrame(model.means_, columns=["ret20", "vol", "ret60", "zscore"])
    remaining = means.copy()

    if feat_mean is not None and feat_std is not None:
        real_ret20 = remaining["ret20"] * feat_std["ret20"] + feat_mean["ret20"]
    else:
        logger.warning("label_states: no feat_mean/feat_std supplied, ranking on standardised means")
        real_ret20 = remaining["ret20"]

    labels = {}

    bear_candidates = real_ret20[real_ret20 < BEAR_RET20_THRESHOLD] if BEAR_ENABLED else real_ret20.iloc[0:0]
    if not bear_candidates.empty:
        bear_idx = bear_candidates.idxmin()
        labels[bear_idx] = "bear"
        real_ret20 = real_ret20.drop(index=bear_idx)
        remaining = remaining.drop(index=bear_idx)

    bull_candidates = real_ret20[real_ret20 > BULL_RET20_THRESHOLD]
    if not bull_candidates.empty:
        bull_idx = bull_candidates.idxmax()
        labels[bull_idx] = "bull"
        remaining = remaining.drop(index=bull_idx)

    for idx in remaining.index:
        labels[idx] = "sideways"

    return labels


# Independent realised-volatility signal (not folded into the regime label).
HIGH_VOL_WINDOW = 20        # days, realised-vol estimation window
HIGH_VOL_LOOKBACK = 252     # days, trailing window for the percentile
HIGH_VOL_PERCENTILE = 0.85  # top 15% of a ticker's own trailing vol counts as high

# Bear-regime position sizing dampener (throttles size rather than
# blocking entries outright). Shared by backtest_engine and run_full_backtest.
BEAR_SIZE_DAMPEN = 0.50

# Regime refit cadence for the walk-forward detector, ~1 trading quarter.
REGIME_REFIT_INTERVAL = 63


def compute_realized_vol(closes, window=HIGH_VOL_WINDOW):
    """Annualised realised volatility from daily returns, rolling `window` days."""
    daily_ret = closes.pct_change()
    return daily_ret.rolling(window).std() * np.sqrt(252)


def compute_high_vol_series(price_df, ticker, window=HIGH_VOL_WINDOW, lookback=HIGH_VOL_LOOKBACK, percentile=HIGH_VOL_PERCENTILE):
    """Boolean Series flagging bars where realised vol exceeds its own
    trailing percentile — relative to the ticker's own history, not a
    cross-ticker cutoff. Causal (backward-looking only), safe for backtests.
    """
    closes = price_df["Close"][ticker].dropna()
    realized_vol = compute_realized_vol(closes, window)
    threshold = realized_vol.rolling(lookback, min_periods=60).quantile(percentile)
    return (realized_vol > threshold).reindex(closes.index).fillna(False)


def is_high_vol_now(price_df, ticker, window=HIGH_VOL_WINDOW, lookback=HIGH_VOL_LOOKBACK, percentile=HIGH_VOL_PERCENTILE):
    """Is the most recent bar high-vol?"""
    flag = compute_high_vol_series(price_df, ticker, window, lookback, percentile)
    return bool(flag.iloc[-1]) if not flag.dropna().empty else False


def build_features(closes):
    """4-feature matrix for HMM fitting: [ret20, vol, ret60, zscore]. No NaN rows."""
    ret20 = closes.pct_change(20)
    vol = closes.pct_change().rolling(20).std()
    ret60 = closes.pct_change(60)
    zscore = (closes.pct_change(20) - closes.pct_change(20).rolling(60).mean()) / \
             (closes.pct_change(20).rolling(60).std() + 1e-8)

    return pd.DataFrame({
        "ret20": ret20,
        "vol": vol,
        "ret60": ret60,
        "zscore": zscore,
    }).dropna()


def _rule_based_fallback(returns):
    """Single-label fallback used only if every HMM restart collapses."""
    mean_r = returns.mean()
    if mean_r > 0.0003:
        return "bull"
    if mean_r < -0.0003 and BEAR_ENABLED:
        return "bear"
    return "sideways"


def detect_regime(price_df, ticker, n_states=3):
    """Fit an HMM on the full price window and label each bar bear/bull/sideways.

    Fits N_RESTARTS random seeds on standardised features and keeps the best
    converged result by log-likelihood, falling back to a rule-based label
    if every restart collapses. n_states is fixed at 3 to match the
    {bear, bull, sideways} vocabulary used downstream.

    Raises ValueError if fewer than HMM_MIN_BARS feature rows are available.
    """
    closes = price_df["Close"][ticker].dropna()
    features = build_features(closes)

    if len(features) < HMM_MIN_BARS:
        raise ValueError(
            f"detect_regime needs >= {HMM_MIN_BARS} feature rows for {ticker}, "
            f"got {len(features)}. Pass a longer price window."
        )

    feat_mean = features.mean()
    feat_std = features.std().replace(0, 1e-8)
    features_scaled = (features - feat_mean) / feat_std

    best_model, best_loglik = None, -np.inf
    best_converged, best_conv_loglik = None, -np.inf
    n_valid = 0

    for i in range(N_RESTARTS):
        fit = _fit_one(features_scaled, n_states, BASE_SEED + i)
        if fit is None:
            continue
        model, converged, loglik = fit
        n_valid += 1
        if loglik > best_loglik:
            best_model, best_loglik = model, loglik
        if converged and loglik > best_conv_loglik:
            best_converged, best_conv_loglik = model, loglik

    if best_converged is not None:
        model = best_converged
    elif best_model is not None:
        logger.info("%s: no restart fully converged (%d/%d usable) — using best by log-likelihood",
                    ticker, n_valid, N_RESTARTS)
        model = best_model
    else:
        fallback = _rule_based_fallback(features.iloc[:, 0])
        logger.warning("%s: all %d restarts collapsed — rule-based fallback (%s) for entire window",
                        ticker, N_RESTARTS, fallback)
        regime_series = pd.Series(fallback, index=features.index)
        return regime_series, fallback, None

    raw_states = model.predict(features_scaled)
    state_map = label_states(model, feat_mean=feat_mean, feat_std=feat_std)
    regime_series = pd.Series(raw_states, index=features.index).map(state_map)
    current_regime = regime_series.iloc[-1]

    return regime_series, current_regime, model


def detect_regime_walkforward(price_df, ticker, n_states=3, refit_interval=REGIME_REFIT_INTERVAL):
    """Causal, walk-forward regime detection.

    Refits every `refit_interval` bars using only data up to that point, and
    commits labels only for newly-revealed bars, so no bar's label depends
    on future data. Widen refit_interval to trade reaction speed for runtime.

    Returns a regime_series with the same index as detect_regime()'s second
    return value.
    """
    closes = price_df["Close"][ticker].dropna()
    features = build_features(closes)

    if len(features) < HMM_MIN_BARS:
        raise ValueError(
            f"detect_regime_walkforward needs >= {HMM_MIN_BARS} feature rows "
            f"for {ticker}, got {len(features)}. Pass a longer price window."
        )

    labels_out = pd.Series(index=features.index, dtype=object)
    checkpoint = HMM_MIN_BARS
    n_refits = 0

    while True:
        window = features.iloc[:checkpoint]
        feat_mean = window.mean()
        feat_std = window.std().replace(0, 1e-8)
        window_scaled = (window - feat_mean) / feat_std

        best_model, best_loglik = None, -np.inf
        best_converged, best_conv_loglik = None, -np.inf
        for i in range(N_RESTARTS):
            fit = _fit_one(window_scaled, n_states, BASE_SEED + i)
            if fit is None:
                continue
            model, converged, loglik = fit
            if loglik > best_loglik:
                best_model, best_loglik = model, loglik
            if converged and loglik > best_conv_loglik:
                best_converged, best_conv_loglik = model, loglik
        model = best_converged if best_converged is not None else best_model
        n_refits += 1

        new_start = 0 if checkpoint == HMM_MIN_BARS else checkpoint - refit_interval
        new_idx = window.index[new_start:checkpoint]

        if model is None:
            fallback = _rule_based_fallback(window.iloc[:, 0])
            labels_out.loc[new_idx] = fallback
        else:
            raw_states = model.predict(window_scaled)
            state_map = label_states(model, feat_mean=feat_mean, feat_std=feat_std)
            chunk_labels = pd.Series(raw_states, index=window.index).map(state_map)
            labels_out.loc[new_idx] = chunk_labels.loc[new_idx]

        if checkpoint >= len(features):
            break
        checkpoint = min(checkpoint + refit_interval, len(features))

    logger.info("%s: walk-forward regime built (%d causal refits, every %d bars)",
                ticker, n_refits, refit_interval)
    return labels_out


if __name__ == "__main__":
    from datetime import datetime

    from data.price_data import fetch_price_data

    today = datetime.today().strftime("%Y-%m-%d")
    price_df = fetch_price_data(["AAPL", "MSFT"], "2010-01-01", today)

    series, current, model = detect_regime(price_df, "AAPL")

    print("last 10 regimes:\n", series.tail(10))
    print("\ncurrent regime:", current)
    print("\nregime counts:\n", series.value_counts())
    print("\nstate means:\n", pd.DataFrame(model.means_, columns=["ret20", "vol", "ret60", "zscore"]))