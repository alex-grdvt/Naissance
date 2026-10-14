"""Evaluates whether held positions should be swapped for stronger candidates.

find_replacements is main.py's L9, called after sizing and stop-loss; also
used by backtest.simulation.run_portfolio_backtest during the walk-forward
simulation loop.
"""

import logging

from layers.signal_engine import compute_short_score

logger = logging.getLogger(__name__)

SCORE_GAP_THRESHOLD = 10    # candidate must beat holding by at least this
RSI_OVERBOUGHT = 70         # candidate RSI must be below this to qualify long
BOTTOM_PCT = 0.20           # bottom 20% of alpha scores counts as a stale holding
SIGNAL_THRESHOLD = 60       # absolute long/short score cutoff to count as a signal

REGIME_ORDER = {"bear": 0, "sideways": 1, "bull": 2}


def _regime_rank(regime):
    """Ordinal rank of a regime (bear < sideways < bull); unknown regimes rank lowest."""
    return REGIME_ORDER.get(regime, -1)


def _get_rsi(price_df, ticker):
    """14-day RSI for the most recent bar."""
    closes = price_df["Close"][ticker].dropna()
    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses
    return round((100 - (100 / (1 + rs))).iloc[-1], 2)


def _is_stale(ticker, scores, all_scores):
    """True if ticker's score sits in the bottom BOTTOM_PCT of all scores."""
    sorted_scores = sorted(all_scores.values())
    cutoff_idx = int(len(sorted_scores) * BOTTOM_PCT)
    cutoff = sorted_scores[cutoff_idx] if cutoff_idx < len(sorted_scores) else 0
    return scores[ticker]["final_score"] <= cutoff


def _build_candidate_pool(price_df, scores, regimes, held, sentiment_df, as_of_date=None):
    """Score every unheld ticker and keep those clearing an absolute long or short signal."""
    all_long_scores = {t: s["final_score"] for t, s in scores.items()}
    pool = []

    for ticker in scores:
        if ticker in held:
            continue

        long_score = scores[ticker]["final_score"]
        regime = regimes.get(ticker, "unknown")
        short_score = compute_short_score(price_df, ticker, sentiment_df, as_of_date=as_of_date, regime=regime)
        rsi = _get_rsi(price_df, ticker)

        has_long_signal = long_score >= SIGNAL_THRESHOLD
        has_short_signal = short_score >= SIGNAL_THRESHOLD

        if has_long_signal and rsi < RSI_OVERBOUGHT:
            pool.append({"ticker": ticker, "direction": "long", "score": long_score, "rsi": rsi, "regime": regime})
        elif has_short_signal and not has_long_signal:
            pool.append({"ticker": ticker, "direction": "short", "score": short_score, "rsi": rsi, "regime": regime})
        else:
            logger.debug("%s skipped — no clear signal (long:%s short:%s)", ticker, long_score, short_score)

    pool.sort(key=lambda c: c["score"], reverse=True)
    return pool, all_long_scores


def find_replacements(price_df, price_df_long, scores, regimes, portfolio_states, sentiment_df=None, as_of_date=None):
    """Decide whether each held position should be swapped for a candidate.

    A vacated (flat) slot is filled with the top-ranked candidate. An open
    position is replaced if either trigger fires:
      A: candidate beats the holding's score by SCORE_GAP_THRESHOLD or more,
         and is in an equal-or-better regime.
      B: the holding is stale (bottom BOTTOM_PCT by score) and the candidate
         scores higher at all.

    as_of_date: the date this call is FOR. Passed straight through to
    compute_short_score's sentiment lookup so a backtest replaying history
    does not score candidates using sentiment from after the simulated
    date. price_df/price_df_long must themselves already be sliced up to
    as_of_date by the caller — this function trusts whatever history it is
    handed and does not truncate it itself.

    Returns (replacements, candidate_pool) — replacements maps held ticker
    to its replacement candidate dict, or None if it should be held.
    """
    held = {t for t, s in portfolio_states.items() if s.get("position") != "flat"}
    candidate_pool, all_long_scores = _build_candidate_pool(
        price_df, scores, regimes, held, sentiment_df, as_of_date=as_of_date
    )

    replacements = {}

    for ticker in held:
        state = portfolio_states[ticker]

        if state.get("position") == "flat":
            best = candidate_pool[0] if candidate_pool else None
            replacements[ticker] = best
            logger.info("%s vacated -> candidate: %s", ticker, best["ticker"] if best else "none")
            continue

        holding_score = scores.get(ticker, {}).get("final_score", 0)
        holding_regime = regimes.get(ticker, "unknown")
        stale = _is_stale(ticker, scores, all_long_scores)

        for candidate in candidate_pool:
            score_gap = candidate["score"] - holding_score
            trigger_a = score_gap >= SCORE_GAP_THRESHOLD and _regime_rank(candidate["regime"]) >= _regime_rank(holding_regime)
            trigger_b = stale and candidate["score"] > holding_score

            if trigger_a or trigger_b:
                replacements[ticker] = candidate
                reason = "A+B" if (trigger_a and trigger_b) else ("A" if trigger_a else "B")
                logger.info(
                    "%s (score:%s) -> %s (score:%s, dir:%s, gap:%.1f, trigger:%s)",
                    ticker, holding_score, candidate["ticker"], candidate["score"],
                    candidate["direction"], score_gap, reason,
                )
                break
        else:
            replacements[ticker] = None
            logger.info("%s held — no replacement qualifies", ticker)

    return replacements, candidate_pool


if __name__ == "__main__":
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from data.factor_data import fetch_factor_data
    from data.macro_data import fetch_macro_data
    from data.price_data import fetch_price_data
    from layers.alpha_score import score_all_tickers
    from layers.kill_switch import check_kill_switch
    from layers.regime_detection import detect_regime
    from portfolio_state import get_state, initialise_db

    today = datetime.today().strftime("%Y-%m-%d")
    signal_start = (datetime.today() - relativedelta(months=13)).strftime("%Y-%m-%d")
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]

    price_df_long = fetch_price_data(tickers, "2020-01-01", today)
    price_df = fetch_price_data(tickers, signal_start, today)
    factor_df = fetch_factor_data("2020-01-01", today)
    macro_df = fetch_macro_data(signal_start, today)

    macro_mode = check_kill_switch(macro_df)
    regimes = {}
    for ticker in tickers:
        _, current_regime, _ = detect_regime(price_df_long, ticker)
        regimes[ticker] = current_regime

    scores = score_all_tickers(price_df, price_df_long, factor_df, None, regimes, macro_mode)

    initialise_db()
    portfolio_states = {t: get_state(t) for t in tickers}

    replacements, pool = find_replacements(price_df, price_df_long, scores, regimes, portfolio_states)
    print("\nCandidate pool (top 5):")
    for c in pool[:5]:
        print(f'  {c["ticker"]} | {c["direction"]} | score: {c["score"]} | rsi: {c["rsi"]}')