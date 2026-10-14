"""Naissance's L6: decides BUY / SHORT / EXIT / HOLD / BLOCK / CASH per ticker
using the entry and exit gates in trade_logic.py.

Correlation-flagged pairs, the 'neutral' macro stance, bear-regime dampening and
high_vol do not gate entries here. Stop-loss exits are handled separately in L8
(layers/stop_loss.py), which main.py calls after this layer.

Gate inputs use factor_adjusted_score rather than the macro-scaled final_score,
because trade_logic's SCORE_GAP is calibrated against factor_adjusted_score.
"""
import pandas as pd
from portfolio_state import get_state, update_state, initialise_db
from layers.trade_logic import (
    long_entry_gates, short_entry_gates,
    long_exit_due_to_regime, short_exit_due_to_regime,
    compute_rsi, compute_rsi_roc,
)


def make_decision(regime, long_score, short_score, rsi, rsi_roc, macro_mode,
                   position, enable_shorts=False):
    """
    Decides the action for one ticker on the current bar: an exit check if a
    position is already held, otherwise long_entry_gates/short_entry_gates
    for a flat ticker. See the module docstring for what is deliberately
    excluded from gating here.
    """
    if macro_mode == 'cash':
        return 'EXIT' if position in ('long', 'short') else 'CASH'

    if position == 'long':
        return 'EXIT' if long_exit_due_to_regime(regime) else 'HOLD'
    if position == 'short':
        return 'EXIT' if short_exit_due_to_regime(regime) else 'HOLD'

    # flat: long and short entry gates are mutually exclusive when
    # SCORE_GAP > 0, since long_score > short_score + GAP and
    # short_score > long_score + GAP cannot both hold
    wants_long, _  = long_entry_gates(long_score, short_score, rsi, rsi_roc, regime)
    wants_short, _ = short_entry_gates(long_score, short_score, rsi, rsi_roc, regime,
                                        enable_shorts=enable_shorts)

    if wants_long:
        # a short macro stance drops an otherwise-valid long signal entirely
        # rather than downgrading it
        return 'BLOCK' if macro_mode == 'short' else 'BUY'
    if wants_short:
        return 'SHORT'
    return 'HOLD'


def run_decision_matrix(price_df, scores, regimes, macro_mode, flagged_pairs=None,
                         enable_shorts=False):
    """
    flagged_pairs is accepted for interface compatibility but is not used;
    see the module docstring for why correlation does not gate entries here.
    enable_shorts defaults to False, matching run_full_backtest.py's default;
    pass True to opt in, as with that script's --shorts flag.
    """
    initialise_db()
    decisions = {}

    for ticker, breakdown in scores.items():
        regime = regimes.get(ticker, 'unknown')
        # NOT final_score — see module docstring.
        long_score = breakdown.get('factor_adjusted_score', breakdown['final_score'])
        short_score = breakdown.get('short_score') or 0

        closes = price_df['Close'][ticker].dropna()
        rsi = compute_rsi(closes).iloc[-1]
        rsi_roc = compute_rsi_roc(closes).iloc[-1]
        if pd.isna(rsi):
            rsi = 50.0
        if pd.isna(rsi_roc):
            rsi_roc = 0.0

        state = get_state(ticker)
        position = state['position']

        action = make_decision(regime, long_score, short_score, rsi, rsi_roc,
                                macro_mode, position, enable_shorts=enable_shorts)

        update_state(ticker, current_regime=regime)

        decisions[ticker] = {
            'action': action,
            'alpha_score': long_score,
            'short_score': short_score,
            'regime': regime,
            'rsi': round(rsi, 2),
            'rsi_roc': round(rsi_roc, 2),
        }

        print(f"\n{ticker} | {action} | long: {long_score} | short: {short_score} | "
              f"regime: {regime} | rsi: {round(rsi, 2)} | rsi_roc: {round(rsi_roc, 2)}")

    return decisions


if __name__ == '__main__':
    from data.price_data import fetch_price_data
    from data.macro_data import fetch_macro_data
    from data.factor_data import fetch_factor_data
    from layers.kill_switch import check_kill_switch
    from layers.regime_detection import detect_regime
    from layers.alpha_score import score_all_tickers
    from layers.correlation_monitor import run_correlation_monitor
    from portfolio_state import update_regimes
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    today = datetime.today().strftime('%Y-%m-%d')
    signal_start = (datetime.today() - relativedelta(months=13)).strftime('%Y-%m-%d')
    regime_start = '2020-01-01'
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']

    price_df_long = fetch_price_data(tickers, regime_start, today)
    price_df = fetch_price_data(tickers, signal_start, today)
    factor_df = fetch_factor_data(regime_start, today)
    macro_df = fetch_macro_data(signal_start, today)

    macro_mode = check_kill_switch(macro_df)
    print("macro mode:", macro_mode)

    regimes = {}
    for ticker in tickers:
        _, current_regime, _ = detect_regime(price_df_long, ticker)
        regimes[ticker] = current_regime

    update_regimes(regimes)  # log regime changes to DB before decisions

    scores = score_all_tickers(price_df, price_df_long, factor_df, None, regimes, macro_mode)
    _, flagged_pairs, _ = run_correlation_monitor(price_df, tickers)
    decisions = run_decision_matrix(price_df, scores, regimes, macro_mode, flagged_pairs)