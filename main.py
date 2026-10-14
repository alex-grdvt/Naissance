"""
Orchestrates the daily Naissance pipeline across ten layers: macro kill-switch,
regime detection, alpha scoring, backtesting, correlation checks, decision
making, position sizing, stop-loss, replacement logic, and execution.

Layer map
---------
L1   kill_switch          -> macro_mode in {long, neutral, short, cash}
L2   regime_detection     -> regimes {ticker: label}
L3   alpha_score          -> scores {ticker: breakdown} including short_score
L4   backtest_engine      -> populates DB with trade history / Kelly inputs
L5   correlation_monitor  -> flagged_pairs, avg_corr
L6   decision_matrix      -> decisions {ticker: action}
L7   asset_sizing         -> sizes {ticker: fraction}
L8   stop_loss            -> stop-adjusted decisions
L9   replacement_logic    -> replacements {stale_ticker: candidate}
L10  execution            -> log_trade + update_state per action
"""

import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

from data.price_data    import fetch_price_data
from data.macro_data    import fetch_macro_data
from data.factor_data   import fetch_factor_data
from data.sentiment_data import fetch_sentiment_data

import universe

from layers.kill_switch         import check_kill_switch
from layers.regime_detection    import detect_regime
from layers.alpha_score         import score_all_tickers
from layers.backtest_engine     import run_backtest
from layers.correlation_monitor import run_correlation_monitor
from layers.decision_matrix     import run_decision_matrix
from asset_sizing import run_single_asset, run_multi_asset
from layers.stop_loss           import run_stop_loss
from layers.replacement_logic   import find_replacements

from portfolio_state import (
    initialise_db,
    update_regimes,
    update_state,
    log_trade,
    log_signal,
    get_state,
)

# Universe

EQUITY_TICKERS = universe.TICKERS

REGIME_START  = '2020-01-01'
SIGNAL_WINDOW = 13   # months of price history for signal features

# Minimum backtest Sharpe required for a ticker to be tradable
SHARPE_GATE = 0.5

# Helpers

def _date_range():
    """Returns (today, signal_start) as 'YYYY-MM-DD' strings; signal_start is SIGNAL_WINDOW
    months earlier.
    """
    today        = datetime.today().strftime('%Y-%m-%d')
    signal_start = (datetime.today() - relativedelta(months=SIGNAL_WINDOW)).strftime('%Y-%m-%d')
    return today, signal_start


def _portfolio_value(portfolio_states):
    """
    Sums position notional values across all tickers in the DB state.
    Returns 10,000 as a default when no live positions exist yet.
    """
    total = sum(
        s.get('position_size', 0) or 0
        for s in portfolio_states.values()
    )
    return total if total > 0 else 10_000.0


def _sharpe_ok(ticker):
    """Returns True if the ticker's latest backtest Sharpe clears SHARPE_GATE, or if no
    backtest exists yet.
    """
    from portfolio_state import get_backtest_results
    rows = get_backtest_results(ticker)   # most recent run first, empty if none
    if rows is None or rows.empty:
        return True   # no backtest data, do not block
    sharpe = rows.iloc[0]['sharpe']
    return (sharpe or 0) >= SHARPE_GATE


# Pipeline

def run():
    """Runs the daily pipeline (L1-L10) once: fetches data, scores and sizes positions,
    applies stops and replacements, and records paper trades.
    """
    print('=' * 60)
    print(f'Naissance daily run  {datetime.today().strftime("%Y-%m-%d %H:%M")}')
    print('sizing mode: single-ticket all-in / multi-ticket diversified (decided per day at L7)')
    print('=' * 60)

    initialise_db()
    today, signal_start = _date_range()

    # data
    print('\n[DATA] fetching …')
    price_df_long = fetch_price_data(EQUITY_TICKERS, REGIME_START, today)
    price_df      = fetch_price_data(EQUITY_TICKERS, signal_start, today)
    macro_df      = fetch_macro_data(signal_start, today)
    factor_df     = fetch_factor_data(signal_start, today)
    sentiment_df  = fetch_sentiment_data(EQUITY_TICKERS, signal_start, today)

    # factor_df only covers SIGNAL_WINDOW; the backtest needs 5Y, hence a
    # separate fetch over REGIME_START.
    backtest_factor_df = fetch_factor_data(REGIME_START, today)

    # L1: kill switch
    print('\n[L1] kill switch')
    macro_mode = check_kill_switch(macro_df)
    print(f'  macro_mode: {macro_mode}')

    if macro_mode == 'cash':
        print('  CASH mode — no new positions. Exiting.')
        return

    # L2: regime detection
    print('\n[L2] regime detection')
    regimes = {}
    for ticker in EQUITY_TICKERS:
        try:
            _, current_regime, _ = detect_regime(price_df_long, ticker)
            regimes[ticker] = current_regime
            print(f'  {ticker}: {current_regime}')
        except Exception as e:
            print(f'  {ticker}: regime detection failed — {e}')
            regimes[ticker] = 'sideways'   # safe fallback

    update_regimes(regimes)

    # L3: alpha scoring (long + short)
    print('\n[L3] alpha scoring')
    scores = score_all_tickers(
        price_df, price_df_long, factor_df, sentiment_df, regimes, macro_mode
    )
    if not scores:
        print('  no scores returned — aborting.')
        return

    # log signals
    for ticker, bd in scores.items():
        log_signal(
            ticker       = ticker,
            alpha_score  = bd['final_score'],
            raw_score    = bd['raw_score'],
            factor_alpha = bd.get('factor_alpha'),
            regime       = bd['regime'],
            macro_mode   = macro_mode,
        )

    # L4: backtest (skips a ticker internally if its DB history is already populated)
    print('\n[L4] backtest')
    for ticker in EQUITY_TICKERS:
        result = run_backtest(ticker, backtest_factor_df)
        if result:
            sharpe = result.get('sharpe', 0)
            print(f'  {ticker}: Sharpe(ann)={sharpe:.2f}')
            if sharpe < SHARPE_GATE:
                print(f'  {ticker}: Sharpe below gate ({SHARPE_GATE}) — will be blocked at L6')

    # L5: correlation monitor
    print('\n[L5] correlation monitor')
    holdings = [t for t in EQUITY_TICKERS if get_state(t).get('position') not in (None, 'flat', '')]
    if len(holdings) >= 2:
        corr_matrix, flagged_pairs, avg_corr = run_correlation_monitor(price_df, holdings)
        print(f'  avg portfolio corr: {avg_corr:.2f}')
        if flagged_pairs:
            print(f'  flagged pairs: {flagged_pairs}')
    else:
        flagged_pairs = []
        print('  < 2 holdings — skipping correlation check')

    # L6: decision matrix
    print('\n[L6] decision matrix')

    # restrict to tickers that cleared the Sharpe gate
    gated_scores = {t: bd for t, bd in scores.items() if _sharpe_ok(t)}
    blocked      = set(scores) - set(gated_scores)
    if blocked:
        print(f'  Sharpe-gated (blocked): {blocked}')

    decisions = run_decision_matrix(
        price_df, gated_scores, regimes, macro_mode, flagged_pairs
    )

    # L7: position sizing
    # run_multi_asset ranks actionable tickers by expected edge and water-fills
    # against per-ticker, category and asset-class caps and a per-ticker Kelly ceiling.
    print('\n[L7] portfolio optimizer')
    # portfolio_states: {ticker: {field: value}}, used by run_stop_loss,
    # find_replacements and _portfolio_value below.
    portfolio_states = {t: get_state(t) for t in EQUITY_TICKERS}
    portfolio_value  = _portfolio_value(portfolio_states)

    actionable = [t for t, d in decisions.items() if d['action'] in ('BUY', 'SHORT')]
    sizes = {}
    if len(actionable) == 1:
        # single actionable ticker: always fully deployed (100%, 50% if
        # high-vol) — see asset_sizing.run_single_asset docstring
        print('  single actionable ticker -- sizing via run_single_asset (all-in)')
        sizes = run_single_asset(actionable[0], portfolio_value, price_df=price_df_long)
        for ticker, frac in sizes.items():
            notional = frac * portfolio_value
            print(f'  {ticker}: {frac:.1%} of portfolio = £{notional:,.0f}')
    elif actionable:
        sizes = run_multi_asset(actionable, gated_scores, regimes, decisions,
                                 portfolio_value, price_df=price_df_long)
        for ticker, frac in sizes.items():
            notional = frac * portfolio_value
            print(f'  {ticker}: {frac:.1%} of portfolio = £{notional:,.0f}')
    else:
        print('  no actionable signals')

    # L8: stop loss
    print('\n[L8] stop loss')
    # run_stop_loss flips decisions[ticker]['action'] to 'EXIT' internally
    # wherever triggered; this just surfaces a summary.
    stop_actions, decisions = run_stop_loss(price_df, decisions, regimes, portfolio_states)
    n_triggered = sum(1 for s in stop_actions.values() if s.get('triggered'))
    print(f'  {n_triggered} stop(s) triggered' if n_triggered else '  no stops triggered')

    # L9: replacement logic
    print('\n[L9] replacement logic')
    replacements = find_replacements(
        price_df, price_df_long, scores, regimes, portfolio_states, sentiment_df
    )
    if replacements:
        for old, new in replacements.items():
            print(f'  replace {old} → {new}')
    else:
        print('  no replacements needed')

    # L10: execution (paper)
    print('\n[L10] execution')
    today_str = datetime.today().strftime('%Y-%m-%d')

    for ticker, decision in decisions.items():
        action     = decision['action']
        regime     = decision.get('regime', regimes.get(ticker, 'unknown'))
        score      = decision.get('alpha_score')
        state      = get_state(ticker)
        position   = state.get('position')
        entry_price= state.get('entry_price')

        try:
            current_price = float(price_df['Close'][ticker].dropna().iloc[-1])
        except Exception:
            print(f'  {ticker}: could not read current price, skipping')
            continue

        if action == 'BUY' and position not in ('long',):
            frac = sizes.get(ticker, 0)
            if frac <= 0:
                print(f'  {ticker}: BUY signal but zero size — skipping')
                continue
            log_trade(
                ticker=ticker, action='ENTRY', position='long',
                entry_price=current_price, exit_price=None,
                entry_date=today_str, regime=regime,
                alpha_score=score, source='live',
            )
            update_state(ticker,
                position='long',
                entry_price=current_price,
                position_size=frac * portfolio_value,
                current_regime=regime,
            )
            print(f'  {ticker}: ENTRY long @ {current_price:.2f}  size={frac:.1%}')

        elif action == 'SHORT' and position not in ('short',):
            frac = sizes.get(ticker, 0)
            if frac <= 0:
                print(f'  {ticker}: SHORT signal but zero size — skipping')
                continue
            log_trade(
                ticker=ticker, action='ENTRY', position='short',
                entry_price=current_price, exit_price=None,
                entry_date=today_str, regime=regime,
                alpha_score=score, source='live',
            )
            update_state(ticker,
                position='short',
                entry_price=current_price,
                position_size=frac * portfolio_value,
                current_regime=regime,
            )
            print(f'  {ticker}: ENTRY short @ {current_price:.2f}  size={frac:.1%}')

        elif action == 'EXIT' and position in ('long', 'short'):
            log_trade(
                ticker=ticker, action='EXIT', position=position,
                entry_price=entry_price, exit_price=current_price,
                entry_date=state.get('entry_date', today_str), regime=regime,
                alpha_score=score, source='live', exit_date=today_str,
            )
            update_state(ticker,
                position='flat',
                entry_price=None,
                position_size=0,
                current_regime=regime,
            )
            print(f'  {ticker}: EXIT {position} @ {current_price:.2f}')

        elif action in ('HOLD', 'BLOCK', 'REDUCE', 'CASH'):
            print(f'  {ticker}: {action}')

    # replacement execution
    for old_ticker, new_ticker in replacements.items():
        old_state = get_state(old_ticker)
        if old_state.get('position') in ('long', 'short'):
            try:
                old_price = float(price_df['Close'][old_ticker].dropna().iloc[-1])
            except Exception:
                continue
            log_trade(
                ticker=old_ticker, action='EXIT', position=old_state['position'],
                entry_price=old_state.get('entry_price'), exit_price=old_price,
                entry_date=old_state.get('entry_date', today_str),
                regime=regimes.get(old_ticker, 'unknown'),
                alpha_score=None, source='live', exit_date=today_str,
            )
            update_state(old_ticker, position='flat', entry_price=None, position_size=0)
            print(f'  [REPLACE] exited {old_ticker}')

        try:
            new_price = float(price_df['Close'][new_ticker].dropna().iloc[-1])
            frac      = sizes.get(new_ticker, 0.05)
            log_trade(
                ticker=new_ticker, action='ENTRY', position='long',
                entry_price=new_price, exit_price=None,
                entry_date=today_str, regime=regimes.get(new_ticker, 'unknown'),
                alpha_score=scores.get(new_ticker, {}).get('final_score'),
                source='live',
            )
            update_state(new_ticker,
                position='long',
                entry_price=new_price,
                position_size=frac * portfolio_value,
                current_regime=regimes.get(new_ticker, 'unknown'),
            )
            print(f'  [REPLACE] entered {new_ticker} @ {new_price:.2f}')
        except Exception as e:
            print(f'  [REPLACE] failed to enter {new_ticker}: {e}')

    print('\n' + '=' * 60)
    print('Run complete.')
    print('=' * 60)


if __name__ == '__main__':
    run()