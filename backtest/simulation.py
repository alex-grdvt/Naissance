"""backtest/simulation.py — concurrent, shared-capital-pool walk-forward engine.

  _load_universe          loads every ticker onto a shared trading calendar
  _build_slice            builds a per-ticker, per-date price slice
  run_portfolio_backtest  the simulation loop
  _clear_previous         removes earlier backtest rows before a re-run

The ticker list, starting capital and target number of concurrent positions are
explicit parameters of run_portfolio_backtest; the defaults mirror the constants
in run_full_backtest.py. MIN_WARMUP, SENTIMENT_CONFIRMATION_MIN_TRADES,
CORR_THROTTLE_PAIR / CORR_THROTTLE_AVG and KILL_SWITCH_NEUTRAL_THROTTLE are
tuning constants internal to this module.
"""

import io
import contextlib
import sqlite3
from collections import Counter
import numpy as np
import pandas as pd

from portfolio_state import DB_PATH, log_trade

from layers.backtest_utils import (
    BARS_PER_DAY,
    _load_prices,
    _stop_hit_long,
    _stop_hit_short,
    _walk_forward_kelly_size,
    ROUND_TRIP_COST_BPS,
    _round_trip_cost_pct,
)
# Gate constants and the entry decision come from layers.trade_logic, the single
# source of the entry and exit rules.
from layers.trade_logic import (
    compute_rsi as _compute_rsi,
    compute_rsi_roc as _compute_rsi_roc,
    SCORE_GAP,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    FAST_MOM_THRESH,
    FAST_MOM_THRESH_SHORT,
    long_entry_gates,
    short_entry_gates,
)
# This module does not import layers.backtest_engine, so the portfolio backtest
# runs without it.
from layers.regime_detection import (
    detect_regime, detect_regime_walkforward, compute_high_vol_series, BEAR_SIZE_DAMPEN
)

from layers.signal_engine import compute_alpha_score, compute_short_score
from layers.factor_decomposition import apply_factor_penalty_both
from layers.correlation_monitor import (
    compute_correlation_matrix,
    flag_high_correlations,
    get_average_correlation,
    apply_direction_signs,
    PAIR_THRESHOLD,
    AVERAGE_THRESHOLD,
)
from layers.kill_switch import check_kill_switch
from layers.regime_confirmation import confirm_regime
from layers.position_sizing import CASH_FLOOR
# Sizing is dispatched on each simulated day by candidate count, as in main.py's L7:
# one candidate goes through asset_sizing.run_single_asset (fully deployed), two or
# more through asset_sizing.run_multi_asset (diversified). See SIZING DISPATCH below.
#
# find_replacements is a pure function that takes portfolio_states as a parameter
# instead of reading the database, so it is safe to call in a historical simulation.
# Its as_of_date is set in the replacement block of run_portfolio_backtest to avoid
# sentiment look-ahead.
from layers.replacement_logic import find_replacements


MIN_WARMUP       = 300     # bars — mirrors backtest_engine._simulate

# Sentiment confirmation of regime transitions (layers.regime_confirmation). Below this
# many closed trades for a ticker, its regime is the raw HMM label. At or above it, a
# transition must agree with recent sentiment or the prior regime holds for another
# cycle. Matches the sentiment_active threshold in backtest_engine._resolve_maturity.
SENTIMENT_CONFIRMATION_MIN_TRADES = 50

# Correlation throttle: fraction of position size kept when an entry is flagged.
CORR_THROTTLE_PAIR = 0.5   # a specific pair (this candidate + a holding) is flagged -- cut hard
CORR_THROTTLE_AVG  = 0.75  # portfolio-wide avg correlation is flagged instead -- lighter cut

# Kill-switch throttle for the macro 'neutral' stance (fraction of size kept, as above).
KILL_SWITCH_NEUTRAL_THROTTLE = 0.5

# Defaults for starting_capital and target_concurrent_positions; they mirror
# STARTING_CAPITAL and TARGET_CONCURRENT_POSITIONS in run_full_backtest.py.
_DEFAULT_STARTING_CAPITAL = 100_000
_DEFAULT_TARGET_CONCURRENT_POSITIONS = 1

# Portfolio-level risk controls.
# kill_switch.py is a macro gate (VIX, yield curve, recession probability) that
# returns a stance of 'cash', 'short', 'neutral' or 'long'; it does not read the
# portfolio's own equity curve. The drawdown throttle in position_sizing.py
# (get_portfolio_drawdown with DRAWDOWN_MILD / DRAWDOWN_SEVERE) reads live
# trade_log rows only.
#
# This simulation applies both: the macro kill switch as written, and a drawdown
# throttle computed from the simulated equity curve using position_sizing's
# thresholds and reductions.


def _clear_previous(tickers):
    """Deletes earlier 'backtest' trades and results for tickers, plus any PORTFOLIO summary
    row, so a re-run starts clean.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ','.join('?' for _ in tickers)
    cur.execute(f"DELETE FROM trade_log WHERE ticker IN ({placeholders}) AND source = 'backtest'", tickers)
    cur.execute(f"DELETE FROM backtest_results WHERE ticker IN ({placeholders})", tickers)
    # Also clear any previous PORTFOLIO-level summary row.
    cur.execute("DELETE FROM backtest_results WHERE ticker = 'PORTFOLIO'")
    conn.commit()
    conn.close()


# ── Universe loading ──────────────────────────────────────────────────────────

def _load_universe(tickers, start, end, interval='1d', period=None):
    """Loads each ticker's OHLCV history independently via layers.backtest_utils
    (so caching matches the single-ticker engine) and builds:
      price_dict[ticker]      the ticker's own full price_df
      full_daily_ret[ticker]  full-history percentage change of close
      shared_dates            trading days common to all tickers, after each
                              ticker's own MIN_WARMUP
      daily_return_df         combined MultiIndex frame [('daily_return', ticker), ...]
                              aligned to shared_dates, for correlation_monitor

    interval / period are passed straight to _load_prices. main() sets period for
    any non-daily interval, because Yahoo limits intraday history to 60 days.
    """
    price_dict     = {}
    full_daily_ret = {}
    date_indexes   = []

    for ticker in tickers:
        price_df = _load_prices(ticker, start, end, interval=interval, period=period)
        if price_df is None or price_df.empty:
            print(f"  [UNIVERSE] {ticker} — price fetch failed, dropping from universe")
            continue
        closes = price_df['Close'][ticker].dropna()
        if len(closes) < MIN_WARMUP + 1:
            print(f"  [UNIVERSE] {ticker} — insufficient history ({len(closes)} bars), dropping")
            continue
        price_dict[ticker]     = price_df
        full_daily_ret[ticker] = closes.pct_change()
        date_indexes.append(closes.index)

    if not price_dict:
        raise RuntimeError("no tickers in universe survived loading")

    shared_dates = date_indexes[0]
    for idx in date_indexes[1:]:
        shared_dates = shared_dates.intersection(idx)
    shared_dates = shared_dates.sort_values()

    if len(shared_dates) <= MIN_WARMUP:
        raise RuntimeError("shared calendar too short after warmup")
    shared_dates = shared_dates[MIN_WARMUP:]

    daily_return_wide = pd.DataFrame({t: full_daily_ret[t] for t in price_dict})
    daily_return_wide = daily_return_wide.reindex(shared_dates)
    daily_return_df = pd.concat({'daily_return': daily_return_wide}, axis=1)

    return price_dict, full_daily_ret, shared_dates, daily_return_df


def _build_slice(price_df, ticker, date, full_daily_ret_series):
    """Builds the per-bar price slice for a ticker, keyed by date rather than integer
    position. Date keys are needed because tickers in the shared loop no longer
    share row positions once missing rows have been dropped independently.
    """
    sub = price_df.loc[:date]
    price_slice = {col: sub[col][ticker] for col in sub.columns.get_level_values(0).unique()}
    slice_df = pd.concat(price_slice, axis=1)
    slice_df.columns = pd.MultiIndex.from_tuples(
        [(c, ticker) for c in price_slice.keys()], names=['Price', 'Ticker']
    )
    slice_df[('daily_return', ticker)] = full_daily_ret_series.loc[:date].values
    return slice_df


# ── Main concurrent simulation ────────────────────────────────────────────────

def run_portfolio_backtest(tickers, start, end, factor_df, sentiment_df=None, macro_df=None,
                            enable_shorts=False, interval='1d', period=None,
                            starting_capital=_DEFAULT_STARTING_CAPITAL,
                            target_concurrent_positions=_DEFAULT_TARGET_CONCURRENT_POSITIONS):
    """Runs the concurrent shared-capital walk-forward simulation across tickers.

    starting_capital and target_concurrent_positions default to the values in
    run_full_backtest.py. Returns a dict with portfolio_trades,
    closed_trades_by_ticker, equity_curve, final_cash, avg_cash_pct, price_dict,
    shared_dates and diagnostics.
    """
    price_dict, full_daily_ret, shared_dates, daily_return_df = \
        _load_universe(tickers, start, end, interval=interval, period=period)
    active_tickers = list(price_dict.keys())

    # REGIME_REFIT_INTERVAL (63 bars) is about one trading quarter of daily bars. For
    # finer intervals it is scaled by the bars-per-day ratio so that a comparable
    # fraction of the dataset lies between refits, and capped so that a short intraday
    # window still gets several refits.
    bars_per_day = BARS_PER_DAY.get(interval, 1)
    regime_refit_interval = min(63 * bars_per_day, max(len(shared_dates) // 10, 1))

    print(f"  [PORTFOLIO] universe: {active_tickers}")
    bar_label = 'bars' if interval != '1d' else 'shared trading days'
    print(f"  [PORTFOLIO] simulating {len(shared_dates)} {bar_label} "
          f"({shared_dates[0].date()} -> {shared_dates[-1].date()})")

    # detect_regime() fits once on the whole window, which leaks future data into
    # regime-gated decisions; detect_regime_walkforward() refits periodically using
    # only the causal prefix. Setting this to False uses the faster but
    # look-ahead-biased detect_regime().
    USE_WALKFORWARD_REGIME = True
    regimes = {}
    # Independent realised-volatility signal, separate from the regime label (see
    # regime_detection.compute_high_vol_series). Its rolling threshold looks backward
    # only, so indexing it bar by bar introduces no look-ahead.
    high_vol_series = {}
    for ticker in active_tickers:
        try:
            if USE_WALKFORWARD_REGIME:
                regimes[ticker] = detect_regime_walkforward(price_dict[ticker], ticker,
                                                             refit_interval=regime_refit_interval)
            else:
                regime_series, _, _ = detect_regime(price_dict[ticker], ticker)
                regimes[ticker] = regime_series
            high_vol_series[ticker] = compute_high_vol_series(price_dict[ticker], ticker)

            # Diagnostic: per ticker, the fraction of bars given each regime label and the mean
            # forward 20-day return on bars carrying each label. A near-zero or positive mean on
            # 'bear' bars suggests the HMM is mislabelling rather than calling a real down-leg;
            # review BEAR_RET20_THRESHOLD (regime_detection.py) or the refit interval.
            _rs = regimes[ticker]
            _fwd_ret20 = price_dict[ticker]['Close'][ticker].pct_change(20).shift(-20)
            _counts = _rs.value_counts(normalize=True)
            print(f"  [BEAR-DIAG] {ticker} regime mix: " +
                  ", ".join(f"{lbl}={pct:.1%}" for lbl, pct in _counts.items()))
            for _lbl in ('bear', 'bull', 'sideways'):
                _mask = _rs == _lbl
                if _mask.any():
                    _real_fwd = _fwd_ret20.reindex(_rs.index)[_mask].mean()
                    print(f"    {_lbl:9s} — mean REAL fwd 20d return while labeled: {_real_fwd:+.2%}")
        except Exception as e:
            print(f"  [PORTFOLIO] {ticker} — regime detection failed, dropping: {e}")

    active_tickers = [t for t in active_tickers if t in regimes]

    positions        = {}   # ticker -> dict(position, entry_price, entry_date, regime, notional, size)
    closed_trades    = {t: [] for t in active_tickers}   # per-ticker pnl history, for walk-forward kelly
    portfolio_trades = []    # every closed trade, any ticker — for reporting
    entry_log = []
    cash = starting_capital
    equity_curve = []        # list of (date_str, total_equity)
    cash_history = []

    # ── Diagnostics: throttle activations and per-ticker entry gating ──────────────
    corr_pair_fires  = 0
    corr_avg_fires   = 0
    kill_neutral_days = 0
    kill_short_days   = 0
    kill_cash_days    = 0
    bear_dampen_fires = 0   # entries taken at reduced size due to bear regime
    regime_confirm_blocks = 0   # sentiment-confirmation held the prior regime instead of the new HMM label
    confirmed_regime_state = {t: None for t in active_tickers}   # last CONFIRMED regime per ticker
    gate_counts = {t: Counter() for t in active_tickers}
    short_gate_counts = {t: Counter() for t in active_tickers}   # only meaningful when enable_shorts

    macro_warned = False
    stale_recession_warned = False   # the staleness warning is shown once, then suppressed
    n_days = len(shared_dates)
    PROGRESS_EVERY = 50   # heartbeat so a long run does not look hung

    for day_idx, date in enumerate(shared_dates):
        date_str = date.strftime('%Y-%m-%d')

        if day_idx % PROGRESS_EVERY == 0 or day_idx == n_days - 1:
            open_now = len(positions)
            print(f"  [PROGRESS] day {day_idx + 1}/{n_days} ({date_str}) — "
                  f"open positions: {open_now} — trades so far: {len(portfolio_trades)}")

        # ── mark-to-market: today's starting equity before any action ──────
        committed  = 0.0
        unrealized = 0.0
        for ticker, pos in positions.items():
            closes = price_dict[ticker]['Close'][ticker]
            if date not in closes.index:
                continue
            current_price = closes.loc[date]
            if pos['position'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
            committed  += pos['notional']
            unrealized += pos['notional'] * (pnl_pct / 100.0)
        total_equity = cash + committed + unrealized
        equity_curve.append((date_str, total_equity))

        if total_equity > 0:
            cash_history.append(cash / total_equity)

        # ── kill switch: macro-based portfolio stance ───────────────────────
        stance = 'long'
        if macro_df is not None and not macro_df.empty:
            macro_date = date.tz_localize(None) if date.tzinfo is not None else date
            macro_slice = macro_df.loc[:macro_date]
            if not macro_slice.empty:
                try:
                    if stale_recession_warned:
                        # Warning already shown once; suppress the repeat. The stance decision still runs
                        # every day.
                        _buf = io.StringIO()
                        with contextlib.redirect_stdout(_buf):
                            stance = check_kill_switch(macro_slice, as_of_date=macro_date)
                    else:
                        stance = check_kill_switch(macro_slice, as_of_date=macro_date)
                        stale_recession_warned = True
                except Exception as e:
                    if not macro_warned:
                        print(f"  [KILL SWITCH] check failed, defaulting to 'long': {e}")
                        macro_warned = True
        elif not macro_warned:
            print("  [KILL SWITCH] no macro data — running with stance='long' throughout")
            macro_warned = True

        if stance == 'neutral':
            kill_neutral_days += 1
        elif stance == 'short':
            kill_short_days += 1
        elif stance == 'cash':
            kill_cash_days += 1

        # ── Sentiment confirmation of regime transitions (layers.regime_confirmation) ──
        # Below SENTIMENT_CONFIRMATION_MIN_TRADES closed trades for a ticker, its regime is
        # the raw walk-forward HMM label. At or above it, a transition away from the last
        # confirmed regime must agree with recent sentiment, or the prior regime holds for
        # another cycle.
        regime_today = {}
        for ticker in active_tickers:
            regime_series = regimes[ticker]
            if date not in regime_series.index:
                continue
            raw_regime = regime_series.loc[date]
            prev_regime = confirmed_regime_state.get(ticker)
            if prev_regime is None or len(closed_trades[ticker]) < SENTIMENT_CONFIRMATION_MIN_TRADES:
                active_regime = raw_regime
            else:
                result = confirm_regime(ticker, prev_regime, raw_regime, sentiment_df, as_of_date=date)
                active_regime = result['active_regime']
                if not result['confirmed'] and result['reason'] != 'no_change':
                    regime_confirm_blocks += 1
            confirmed_regime_state[ticker] = active_regime
            regime_today[ticker] = active_regime

        # ── per-ticker exits (stop loss / regime flip / kill-switch cash) ───
        force_flatten = (stance == 'cash')
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            price_df = price_dict[ticker]
            closes = price_df['Close'][ticker]
            if date not in closes.index:
                continue

            if ticker not in regime_today:
                continue
            regime = regime_today[ticker]
            current_price = closes.loc[date]

            slice_df = _build_slice(price_df, ticker, date, full_daily_ret[ticker])

            if pos['position'] == 'long':
                stop_hit    = _stop_hit_long(slice_df, ticker, regime, pos['entry_price'], entry_date=pd.to_datetime(pos['entry_date']))
                regime_flip = (regime == 'bear')
            else:
                stop_hit    = _stop_hit_short(slice_df, ticker, regime, pos['entry_price'], entry_date=pd.to_datetime(pos['entry_date']))
                regime_flip = (regime == 'bull')

            if stop_hit or regime_flip or force_flatten:
                if pos['position'] == 'long':
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                else:
                    pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100

                net_pnl_pct = pnl_pct - _round_trip_cost_pct(pos['position'])
                dollar_pnl  = pos['notional'] * (net_pnl_pct / 100.0)
                cash       += pos['notional'] + dollar_pnl

                if stop_hit:
                    reason = 'stop_loss'
                elif regime_flip:
                    reason = 'regime_flip'
                else:
                    reason = 'kill_switch_cash'

                trade = {
                    'ticker': ticker, 'position': pos['position'],
                    'entry_price': pos['entry_price'], 'exit_price': current_price,
                    'entry_date': pos['entry_date'], 'exit_date': date_str,
                    'pnl': pnl_pct, 'net_pnl': net_pnl_pct,
                    'notional': pos['notional'], 'regime': pos['regime'],
                    'exit_reason': reason,
                    'size': pos['size'],
                    'kelly_f': pos['kelly_f'],
                }
                closed_trades[ticker].append({'pnl': pnl_pct, 'position': pos['position']})
                portfolio_trades.append(trade)
                log_trade(
                    ticker=ticker, action='EXIT', position=pos['position'],
                    entry_price=pos['entry_price'], exit_price=current_price,
                    entry_date=pos['entry_date'], regime=pos['regime'],
                    alpha_score=None, source='backtest', exit_date=date_str,
                    notional=pos['notional'], dollar_pnl=dollar_pnl,
                )
                print(
                    f"EXIT {ticker} | "
                    f"{reason} | "
                    f"entry={pos['entry_date']} | "
                    f"exit={date_str} | "
                    f"regime={regime} | "
                    f"pnl={pnl_pct:.2f}%"
                    )
                del positions[ticker]

        if force_flatten:
            continue  # kill switch says go to cash — no new entries today

        open_tickers = list(positions.keys())

        # ── Per-ticker entries, PASS 1: gather every candidate that clears its gates ──
        # Candidates are collected before any sizing, so that they compete for the capital
        # pool on signal strength (PASS 2) rather than on iteration order over
        # active_tickers.
        candidates = []
        for ticker in active_tickers:
            if ticker in positions:
                continue

            price_df = price_dict[ticker]
            closes = price_df['Close'][ticker]
            if date not in closes.index:
                continue

            if ticker not in regime_today:
                continue
            regime = regime_today[ticker]
            current_price = closes.loc[date]

            slice_df = _build_slice(price_df, ticker, date, full_daily_ret[ticker])

            try:
                raw_long = compute_alpha_score(
                    slice_df, slice_df, ticker, regime, sentiment_df=sentiment_df, as_of_date=date
                )
                raw_short = compute_short_score(slice_df, ticker, sentiment_df=sentiment_df, as_of_date=date, regime=regime)
                try:
                    long_score, short_score, _, _, _ = apply_factor_penalty_both(
                        raw_long, raw_short, slice_df, factor_df, ticker, end_date=date
                    )
                except Exception:
                    long_score, short_score = raw_long, raw_short
            except Exception:
                continue

            closes_upto = closes.loc[:date]
            rsi_series = _compute_rsi(closes_upto)
            rsi = rsi_series.iloc[-1] if not rsi_series.dropna().empty else 50.0
            rsi_roc_series = _compute_rsi_roc(closes_upto)
            rsi_roc = rsi_roc_series.iloc[-1] if not rsi_roc_series.dropna().empty else 0.0

            # ── entry decision — the ONE gate implementation, from trade_logic ──
            wants_long, failed_gates = long_entry_gates(long_score, short_score, rsi, rsi_roc, regime)
            gate_counts[ticker][tuple(failed_gates) if failed_gates else ('pass',)] += 1

            wants_short, failed_gates_short = short_entry_gates(
                long_score, short_score, rsi, rsi_roc, regime, enable_shorts=enable_shorts
            )
            if enable_shorts:
                short_gate_counts[ticker][tuple(failed_gates_short) if failed_gates_short else ('pass',)] += 1

            if not wants_long and not wants_short:
                continue
            # The long and short gates cannot both pass for GAP > 0, so this is either/or.
            direction = 'long' if wants_long else 'short'

            if stance == 'short' and direction == 'long':
                continue
            if stance == 'cash':
                continue   # already flattened above; defensive guard

            # ── Correlation throttle ─────────────────────────────────────────
            # Checked against the positions open at the start of today, not against other
            # candidates from this pass, because PASS 2 has not yet decided which will be
            # funded. Two new same-day entries in correlated names are therefore not caught;
            # SLOT_CAP in PASS 2 limits how much concentration a single day can add.
            corr_throttle = 1.0
            check_set = open_tickers + [ticker]
            if len(check_set) >= 2:
                try:
                    corr_matrix = compute_correlation_matrix(daily_return_df.loc[:date], check_set)
                    directions  = {o: positions[o]['position'] for o in open_tickers}
                    directions[ticker] = direction
                    signed_matrix = apply_direction_signs(corr_matrix, directions)
                    flagged_pairs = flag_high_correlations(signed_matrix)
                    avg_corr      = get_average_correlation(signed_matrix)
                    pair_flagged      = any(ticker in (a, b) for a, b, _ in flagged_pairs)
                    portfolio_flagged = avg_corr > AVERAGE_THRESHOLD
                    if pair_flagged or portfolio_flagged:
                        corr_throttle = CORR_THROTTLE_PAIR if pair_flagged else CORR_THROTTLE_AVG
                        if pair_flagged:
                            corr_pair_fires += 1
                        else:
                            corr_avg_fires += 1
                except Exception:
                    pass  # not enough trailing return history yet — no throttle

            kill_throttle = KILL_SWITCH_NEUTRAL_THROTTLE if stance == 'neutral' else 1.0

            # Sizing (Kelly and diversification caps) is delegated to asset_sizing's
            # run_single_asset / run_multi_asset once PASS 1 has the full candidate set (see the
            # sizing dispatch after this loop). throttle_mult carries the backtest-specific
            # overlays that those functions do not know about: the correlation monitor, the
            # macro kill switch, the portfolio drawdown throttle and bear-regime dampening on
            # long entries.
            throttle_mult = corr_throttle * kill_throttle

            if direction == 'long' and regime == 'bear':
                throttle_mult *= BEAR_SIZE_DAMPEN   # dampens rather than blocks (see wants_long above)
                bear_dampen_fires += 1

            if throttle_mult <= 0:
                continue

            # ── Conviction score: how far this candidate clears its own score gate ──
            # Always > 0 here, since SCORE_GAP was already required to reach this point. The
            # value is in score units, not predicted-return units: it ranks setups on this
            # model's own scale and is a tie-breaker under scarcity, not a return forecast.
            margin = (long_score - short_score - SCORE_GAP) if direction == 'long' \
                     else (short_score - long_score - SCORE_GAP)

            rsi_roc_prev = (
                rsi_roc_series.iloc[-2]
                if len(rsi_roc_series.dropna()) >= 2
                else 0.0
            )

            rsi_acceleration = (
                rsi_roc - rsi_roc_prev
            )

            entry_log.append({
                "date": date_str,
                "ticker": ticker,
                "direction": direction,
                "regime": regime,
                "price": current_price,
                "long_score": long_score,
                "short_score": short_score,
                "rsi": rsi,
                "rsi_roc": rsi_roc,
                "rsi_acceleration": rsi_acceleration,
                "throttle_mult": throttle_mult,
            })
            candidates.append(dict(
                ticker=ticker, direction=direction, regime=regime,
                current_price=current_price, throttle_mult=throttle_mult, margin=margin,
                long_score=long_score, short_score=short_score,
            ))

        # ── SIZING DISPATCH: asset_sizing.run_single_asset vs run_multi_asset ──
        # Per-bet size comes from whichever sizing engine fits today's candidate count, as
        # in main.py's L7: exactly one candidate goes through run_single_asset (fully
        # deployed, no Kelly cap); two or more go through run_multi_asset (diversified,
        # never all-in). PASS 2 below separately decides how much of the shared, partly
        # committed capital pool is free and how to ration it across slots. asset_sizing
        # has no notion of a shared pool, so that part is specific to the backtest.
        if candidates:
            if len(candidates) == 1:
                from asset_sizing import run_single_asset
                only = candidates[0]
                sizes = run_single_asset(only['ticker'], portfolio_value=total_equity, price_df=None)
                base_fraction = sizes.get(only['ticker'], 0.0)
                hv_series = high_vol_series.get(only['ticker'])
                high_vol = bool(hv_series.loc[date]) if hv_series is not None and date in hv_series.index else False
                only['kelly_f'] = base_fraction * only['throttle_mult'] * (0.50 if high_vol else 1.0)
            else:
                from asset_sizing import run_multi_asset
                cand_tickers = [c['ticker'] for c in candidates]
                scores    = {c['ticker']: {'final_score': c['long_score'], 'short_score': c['short_score']}
                             for c in candidates}
                regimes_in = {c['ticker']: c['regime'] for c in candidates}
                decisions_in = {c['ticker']: {'action': 'BUY' if c['direction'] == 'long' else 'SHORT'}
                                for c in candidates}
                # price_df=None: the high-vol halving inside run_multi_asset needs a combined
                # multi-ticker OHLCV frame, which this backtest's per-ticker price_dict does not
                # provide. The halving is applied below instead, from this backtest's own
                # high_vol_series, so the same signal is used without double-counting.
                sizes = run_multi_asset(
                    tickers=cand_tickers, scores=scores, regimes=regimes_in,
                    decisions=decisions_in, portfolio_value=total_equity,
                    price_df=None, run_date=date_str,
                )
                for c in candidates:
                    base_fraction = sizes.get(c['ticker'], 0.0)
                    hv_series = high_vol_series.get(c['ticker'])
                    high_vol = bool(hv_series.loc[date]) if hv_series is not None and date in hv_series.index else False
                    c['kelly_f'] = base_fraction * c['throttle_mult'] * (0.50 if high_vol else 1.0)

            candidates = [c for c in candidates if c['kelly_f'] > 0]

        # ── PASS 2: conviction-weighted allocation ─────────────────────────
        # Each candidate's own half-Kelly ceiling still applies: SLOT_CAP and margin
        # weighting only decide who is squeezed on a crowded day, and `desired` is always
        # capped at kelly_f * total_equity first. TARGET_CONCURRENT_POSITIONS fixes the
        # ceiling per slot, so sizing does not depend on how much cash is idle today.
        #
        # TARGET_CONCURRENT_POSITIONS = min(len(TICKERS), 6). The ceiling of 6 sits within
        # the 3-7 concurrent open positions observed on an 8-ticker universe, and the
        # min() scales it with the universe. Revisit the 6 if TICKERS grows well beyond 8,
        # or if len(candidates) exceeds it on most days rather than only the busy ones.
        #
        # SLOT_CAP always applies. With a single-ticker universe,
        # TARGET_CONCURRENT_POSITIONS is 1, so slot_cap equals total_equity and a lone
        # candidate receives the full allocation from run_single_asset. The cap starts to
        # bind once the universe has more than one ticker.
        if candidates:
            committed_now  = sum(p['notional'] for p in positions.values())
            available_cash = max(total_equity - committed_now - CASH_FLOOR, 0)
            slot_cap       = total_equity / target_concurrent_positions

            for c in candidates:
                desired_uncapped = c['kelly_f'] * total_equity
                c['desired'] = min(desired_uncapped, slot_cap)

            total_desired = sum(c['desired'] for c in candidates)

            if total_desired <= available_cash:
                # plenty of room today — nobody needs to be rationed
                for c in candidates:
                    c['notional'] = c['desired']
            else:
                # Scarce day: split available_cash by conviction margin, never above what a
                # candidate already wanted. Cash that a capped candidate does not use stays
                # uncommitted rather than being redistributed, so one ticker's size does not
                # depend on whether another ticker fired the same day.
                total_margin = sum(c['margin'] for c in candidates)
                for c in candidates:
                    share = (c['margin'] / total_margin) * available_cash
                    c['notional'] = min(c['desired'], share)

            for c in candidates:
                if c['notional'] <= 0:
                    continue
                ticker = c['ticker']
                positions[ticker] = {
                    'position':    c['direction'],
                    'entry_price': c['current_price'],
                    'entry_date':  date_str,
                    'regime':      c['regime'],
                    'notional':    round(c['notional'], 2),
                    # 'size' records the allocation fraction actually deployed after SLOT_CAP and
                    # rationing, not the pre-cap kelly_f, which only gates whether the entry fires.
                    'size':        round(c['notional'] / total_equity, 4),
                    'kelly_f': c['kelly_f'],

                }
                # Capital leaves the free-cash pool on entry, so cash and
                # committed notional stay consistent with total equity.
                cash -= positions[ticker]['notional']
                open_tickers.append(ticker)

        # ── Replacement logic (layers.replacement_logic) ───────────────────
        # Runs after today's entries so that a position opened above is not immediately
        # evaluated as its own replacement. PASS 1 only scores tickers that are not held,
        # so held tickers are scored here too: find_replacements needs every held
        # ticker's current score to judge staleness against the unheld candidate pool.
        if positions:
            scores_today = {c['ticker']: {'final_score': c['long_score']} for c in candidates}
            regimes_today = regime_today   # already computed (confirmed) once for today, above

            for ticker in positions:
                if ticker in scores_today:
                    continue
                price_df = price_dict[ticker]
                closes = price_df['Close'][ticker]
                if date not in closes.index or ticker not in regimes_today:
                    continue
                slice_df = _build_slice(price_df, ticker, date, full_daily_ret[ticker])
                try:
                    raw_long = compute_alpha_score(
                        slice_df, slice_df, ticker, regimes_today[ticker],
                        sentiment_df=sentiment_df, as_of_date=date,
                    )
                    scores_today[ticker] = {'final_score': raw_long}
                except Exception:
                    continue

            # replacement_logic expects one combined multi-ticker OHLCV frame
            # (price_df["Close"][ticker]), not this module's per-ticker price_dict. Build it
            # on demand, sliced to today, for tickers that have a score only.
            scored_tickers = [t for t in scores_today if date in price_dict[t]['Close'][t].index]
            if scored_tickers:
                combined_slice = pd.concat(
                    [price_dict[t].loc[:date] for t in scored_tickers], axis=1
                )
                portfolio_states_today = {
                    t: {'position': positions[t]['position']} for t in positions
                }
                try:
                    replacements, _pool = find_replacements(
                        combined_slice, combined_slice, scores_today, regimes_today,
                        portfolio_states_today, sentiment_df=sentiment_df, as_of_date=date,
                    )
                except Exception:
                    replacements = {}

                for old_ticker, candidate in replacements.items():
                    if not candidate or old_ticker not in positions:
                        continue
                    new_ticker = candidate['ticker']
                    if new_ticker in positions or new_ticker == old_ticker:
                        continue
                    new_price_df = price_dict.get(new_ticker)
                    if new_price_df is None or date not in new_price_df['Close'][new_ticker].index:
                        continue

                    # close old_ticker at today's price, same accounting as
                    # the stop-loss/regime-flip exit path above
                    old_pos = positions.pop(old_ticker)
                    old_close = price_dict[old_ticker]['Close'][old_ticker].loc[date]
                    if old_pos['position'] == 'long':
                        pnl_pct = (old_close - old_pos['entry_price']) / old_pos['entry_price'] * 100
                    else:
                        pnl_pct = (old_pos['entry_price'] - old_close) / old_pos['entry_price'] * 100
                    net_pnl_pct = pnl_pct - _round_trip_cost_pct(old_pos['position'])
                    dollar_pnl  = old_pos['notional'] * (net_pnl_pct / 100.0)
                    cash       += old_pos['notional'] + dollar_pnl
                    closed_trades[old_ticker].append({'pnl': pnl_pct, 'position': old_pos['position']})
                    portfolio_trades.append({
                        'ticker': old_ticker, 'position': old_pos['position'],
                        'entry_price': old_pos['entry_price'], 'exit_price': old_close,
                        'entry_date': old_pos['entry_date'], 'exit_date': date_str,
                        'pnl': pnl_pct, 'net_pnl': net_pnl_pct, 'notional': old_pos['notional'],
                        'regime': old_pos['regime'], 'exit_reason': 'replaced',
                        'kelly_f': old_pos.get('kelly_f'), 'size': old_pos.get('size'),
                    })
                    log_trade(
                        ticker=old_ticker, action='EXIT', position=old_pos['position'],
                        entry_price=old_pos['entry_price'], exit_price=old_close,
                        entry_date=old_pos['entry_date'], regime=old_pos['regime'],
                        alpha_score=None, source='backtest', exit_date=date_str,
                        notional=old_pos['notional'], dollar_pnl=dollar_pnl,
                    )

                    # Open new_ticker with the freed notional (all-in, as in PASS 2 above), capped by
                    # the cash actually free.
                    new_price = price_dict[new_ticker]['Close'][new_ticker].loc[date]
                    notional = min(old_pos['notional'], max(cash, 0))
                    if notional <= 0:
                        continue
                    cash -= notional
                    positions[new_ticker] = {
                        'position':    candidate['direction'],
                        'entry_price': new_price,
                        'entry_date':  date_str,
                        'regime':      candidate.get('regime', regimes_today.get(new_ticker, 'unknown')),
                        'notional':    round(notional, 2),
                        'size':        round(notional / total_equity, 4) if total_equity else 0.0,
                        'kelly_f':     1.0,
                    }
                    open_tickers.append(new_ticker)

    # ── close anything still open at the end of the run ─────────────────────
    last_date = shared_dates[-1]
    for ticker, pos in list(positions.items()):
        closes = price_dict[ticker]['Close'][ticker]
        current_price = closes.loc[last_date]
        if pos['position'] == 'long':
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
        else:
            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
        net_pnl_pct = pnl_pct - _round_trip_cost_pct(pos['position'])
        dollar_pnl  = pos['notional'] * (net_pnl_pct / 100.0)
        cash       += pos['notional'] + dollar_pnl

        trade = {
            'ticker': ticker, 'position': pos['position'], 'entry_price': pos['entry_price'],
            'exit_price': current_price, 'entry_date': pos['entry_date'],
            'exit_date': last_date.strftime('%Y-%m-%d'), 'pnl': pnl_pct, 'net_pnl': net_pnl_pct,
            'notional': pos['notional'], 'regime': pos['regime'], 'exit_reason': 'end_of_backtest','kelly_f': pos['kelly_f'],'size': pos['size'],
        }
        closed_trades[ticker].append({'pnl': pnl_pct, 'position': pos['position']})
        portfolio_trades.append(trade)
        log_trade(
            ticker=ticker, action='EXIT', position=pos['position'],
            entry_price=pos['entry_price'], exit_price=current_price,
            entry_date=pos['entry_date'], regime=pos['regime'],
            alpha_score=None, source='backtest', exit_date=last_date.strftime('%Y-%m-%d'),
            notional=pos['notional'], dollar_pnl=dollar_pnl,
        )
    positions.clear()

    # Exit-reason breakdown across every closure path (stop, regime-flip and kill-switch
    # exits, replacement swaps and forced end-of-run closes), built once from
    # portfolio_trades because 'exit_reason' is recorded on every trade.
    exit_counts = Counter(t['exit_reason'] for t in portfolio_trades)

    pd.DataFrame(entry_log).to_csv(
    "entry_diagnostics.csv",
    index=False
    )
    return {
        'portfolio_trades':       portfolio_trades,
        'closed_trades_by_ticker': closed_trades,
        'equity_curve':           equity_curve,
        'final_cash':             cash,
        'avg_cash_pct': float(np.mean(cash_history)) if cash_history else 0.0,
        'price_dict':             price_dict,
        'shared_dates':           shared_dates,
        'diagnostics': {
            'corr_pair_fires':   corr_pair_fires,
            'corr_avg_fires':    corr_avg_fires,
            'kill_neutral_days': kill_neutral_days,
            'kill_short_days':   kill_short_days,
            'kill_cash_days':    kill_cash_days,
            'regime_confirm_blocks': regime_confirm_blocks,
            'bear_dampen_fires': bear_dampen_fires,
            'gate_counts':       gate_counts,
            'short_gate_counts': short_gate_counts,
            'enable_shorts':     enable_shorts,
            'exit_counts':       exit_counts,
        },
    }