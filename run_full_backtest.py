"""run_full_backtest.py — concurrent, shared-capital-pool backtest of the Naissance universe.

The correlation monitor and kill switch run inside the loop, so their
throttling is tested historically.

Usage:
    python run_full_backtest.py                       # default universe
    python run_full_backtest.py --tickers=AAPL,NVDA   # explicit tickers
    python run_full_backtest.py --universe            # every ticker in universe.py
    python run_full_backtest.py --shorts              # enable short entries
    python run_full_backtest.py --fast                # skip the sentiment fetch
    python run_full_backtest.py --sweep               # SCORE_GAP sensitivity sweep
    python run_full_backtest.py --interval=5m --period=60d

Default universe: five equities plus TLT, GLD and FXE as cross-asset
diversifiers. FXE is used instead of the spot EURUSD=X pair because Yahoo
reports zero volume for spot FX, which would zero the A/D component of
alpha_score.

Earlier source='backtest' rows for the chosen tickers are cleared, one
concurrent simulation runs, and per-ticker and portfolio summaries are printed
alongside Buy & Hold, 12-month ROC momentum and SPY baselines.

Entries use a conviction-weighted two-pass allocation: PASS 1 gathers the
candidates that clear their gates, and PASS 2 allocates cash weighted by
score-gap margin, capped at half-Kelly and a per-slot ceiling (see
TARGET_CONCURRENT_POSITIONS). The engine itself lives in backtest/.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
from dateutil.relativedelta import relativedelta

from portfolio_state import initialise_db, log_backtest_results
from data.factor_data import fetch_factor_data
from data.sentiment_data import fetch_sentiment_data
from data.macro_data import fetch_macro_data

from layers.backtest_utils import BACKTEST_YEARS, BARS_PER_DAY, periods_per_year

from backtest.simulation import run_portfolio_backtest, _clear_previous
from backtest.baselines import _run_buy_and_hold, _run_roc_momentum, _run_spx_benchmark
from backtest.reporting import _portfolio_summary, _ticker_summary, _print_diagnostics, plot_equity_and_drawdown
from backtest.calibration import run_sensitivity_sweep


EQUITY_TICKERS = ['AAPL', 'XOM', 'INTC', 'NVDA', 'TSLA']
DIVERSIFIERS   = ['TLT', 'GLD', 'FXE']   # bonds, gold, FX-proxy ETF
TICKERS        = EQUITY_TICKERS + DIVERSIFIERS

STARTING_CAPITAL = 100_000

# Scales with the universe up to MAX_CONCURRENT_POSITIONS. See the PASS 2 comment
# in backtest/simulation.py for how the ceiling of 6 was chosen.
MAX_CONCURRENT_POSITIONS    = 6
TARGET_CONCURRENT_POSITIONS = min(len(TICKERS), MAX_CONCURRENT_POSITIONS)
SHARPE_GATE      = 0.5     # matches main.py's SHARPE_GATE


def main():
    """Runs the backtest end to end: fetches factor, sentiment and macro data, simulates the
    portfolio, then prints summaries against the baselines and plots equity and drawdown.
    """
    enable_shorts = '--shorts' in sys.argv   # opt-in long+short mode; default is long-only
    print('sizing mode: single-ticket all-in / multi-ticket diversified (decided per day, see backtest/simulation.py)')
    fast_mode = '--fast' in sys.argv   # dev flag: skip sentiment fetch for a quick smoke test

    # Yahoo caps intraday history at 60 days regardless of start/end, so any
    # non-daily interval fetches via period='60d' unless --period overrides it.
    interval = '1d'
    period   = None
    for arg in sys.argv:
        if arg.startswith('--interval='):
            interval = arg.split('=', 1)[1]
        elif arg.startswith('--period='):
            period = arg.split('=', 1)[1]
    if interval != '1d' and period is None:
        period = '60d'

    initialise_db()
    _clear_previous(TICKERS)

    end   = datetime.today().strftime('%Y-%m-%d')
    if interval != '1d':
        # start/end are unused when period is set, but sentiment/macro
        # fetches below still take a range -- approximate it from period.
        period_days = int(''.join(c for c in period if c.isdigit()) or 60)
        start = (datetime.today() - relativedelta(days=period_days)).strftime('%Y-%m-%d')
    else:
        start = (datetime.today() - relativedelta(years=BACKTEST_YEARS)).strftime('%Y-%m-%d')

    # FF5+Mom regression needs a long daily history regardless of price
    # interval, so always pull the full BACKTEST_YEARS of factor data even
    # for a short intraday run.
    factor_start = (datetime.today() - relativedelta(years=BACKTEST_YEARS)).strftime('%Y-%m-%d')
    factor_df = fetch_factor_data(factor_start, end)

    print(f'\n{"="*60}\nSENTIMENT\n{"="*60}')
    if fast_mode:
        print('  [SENTIMENT] --fast mode — skipping fetch, running neutral')
        sentiment_df = None
    else:
        try:
            sentiment_df = fetch_sentiment_data(TICKERS, start, end)
        except Exception as e:
            print(f'  [SENTIMENT] batch fetch failed — running neutral: {e}')
            sentiment_df = None

    print(f'\n{"="*60}\nMACRO (kill switch)\n{"="*60}')
    try:
        macro_df = fetch_macro_data(start, end)
    except Exception as e:
        print(f'  [MACRO] fetch failed — kill switch runs stance=long throughout: {e}')
        macro_df = None

    _ticker_word = 'ticker' if len(TICKERS) == 1 else 'tickers'
    print(f'\n{"="*60}\nPORTFOLIO BACKTEST — {len(TICKERS)} {_ticker_word}, shared ${STARTING_CAPITAL:,.0f} pool\n{"="*60}')
    if enable_shorts:
        print('  [MODE] --shorts passed: long+short entries both enabled')
    result = run_portfolio_backtest(TICKERS, start, end, factor_df, sentiment_df=sentiment_df,
                                     macro_df=macro_df, enable_shorts=enable_shorts,
                                     interval=interval, period=period,
                                     starting_capital=STARTING_CAPITAL,
                                     target_concurrent_positions=TARGET_CONCURRENT_POSITIONS)

    ppy = periods_per_year(interval)
    portfolio_summary = _portfolio_summary(result['equity_curve'], periods_per_year_=ppy)

    print(f'\n\n{"="*90}')
    print(f'{"Ticker":<8} {"Trades":>7} {"WinRate":>8} {"AvgWin":>8} {"AvgLoss":>8} {"$PnL":>14}')
    print('-' * 90)
    for ticker in TICKERS:
        trades_for_ticker = [t for t in result['portfolio_trades'] if t['ticker'] == ticker]
        s = _ticker_summary(trades_for_ticker)
        if s is None:
            print(f'{ticker:<8} {"— no trades —":>45}')
            continue
        print(f"{ticker:<8} {s['n_trades']:>7} {s['win_rate']:>8.1%} "
              f"{s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} {s['dollar_pnl']:>14,.2f}")
    print('=' * 90)

    if enable_shorts:
        # Direction breakdown: a blended long+short line can hide one side
        # quietly bleeding. Long-only runs skip this (would just repeat the table above).
        print(f'\n{"="*90}\nLONG vs SHORT breakdown (--shorts mode)\n{"="*90}')
        print(f'{"Ticker":<8} {"Dir":<6} {"Trades":>7} {"WinRate":>8} {"AvgWin":>8} {"AvgLoss":>8} {"$PnL":>14}')
        print('-' * 90)
        for ticker in TICKERS:
            for direction in ('long', 'short'):
                trades_dir = [t for t in result['portfolio_trades']
                              if t['ticker'] == ticker and t['position'] == direction]
                s = _ticker_summary(trades_dir)
                if s is None:
                    continue
                print(f"{ticker:<8} {direction:<6} {s['n_trades']:>7} {s['win_rate']:>8.1%} "
                      f"{s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} {s['dollar_pnl']:>14,.2f}")
        print('=' * 90)

    if portfolio_summary:
        gate_pass = 'PASS' if portfolio_summary['sharpe'] >= SHARPE_GATE else 'FAIL'
        print(f"\nPORTFOLIO — Sharpe (annualized, daily): {portfolio_summary['sharpe']:.3f} [{gate_pass}] | "
              f"MaxDD: {portfolio_summary['max_drawdown']:.2f}% | "
              f"Final equity: ${portfolio_summary['final_equity']:,.2f} | "
              f"Total return: {portfolio_summary['total_return_pct']:.2f}% | "
              f"CAGR: {portfolio_summary['annualized_return_pct']:.2f}%")

        log_backtest_results(
            ticker='PORTFOLIO', n_trades=len(result['portfolio_trades']),
            win_rate=None, avg_win=None, avg_loss=None,
            sharpe=portfolio_summary['sharpe'], max_drawdown=portfolio_summary['max_drawdown'],
            total_return_pct=portfolio_summary['total_return_pct'],
            annualized_return_pct=portfolio_summary['annualized_return_pct'],
            source='backtest_portfolio',
        )
    else:
        print("\nPORTFOLIO — no trades generated, nothing to summarize")

    # baselines
    bars_per_day = BARS_PER_DAY.get(interval, 1)
    # 252 bars assumes daily bars; a 60-day intraday window does not have 12
    # months of data, so use a 1-week lookback for any intraday interval instead.
    roc_period = 252 if interval == '1d' else bars_per_day * 5
    bh_curve  = _run_buy_and_hold(result['price_dict'], result['shared_dates'], STARTING_CAPITAL)
    roc_curve = _run_roc_momentum(result['price_dict'], result['shared_dates'], STARTING_CAPITAL,
                                   roc_period=roc_period,
                                   rebalance='month' if interval == '1d' else 'day')
    bh_summary  = _portfolio_summary(bh_curve, periods_per_year_=ppy)
    roc_summary = _portfolio_summary(roc_curve, periods_per_year_=ppy)
    try:
        spy_curve = _run_spx_benchmark(result['shared_dates'], STARTING_CAPITAL,
                                        interval=interval, period=period)
        spy_summary = _portfolio_summary(spy_curve, periods_per_year_=ppy)
    except Exception as e:
        print(f'  [BENCHMARK] SPY fetch failed, dropping from comparison: {e}')
        spy_curve, spy_summary = None, None

    print(f'\n{"="*90}\nCOMPARISON — Naissance vs. baselines\n{"="*90}')
    print(f'{"Strategy":<20} {"Sharpe":>8} {"MaxDD":>8} {"TotalRet":>10} {"CAGR":>8} {"FinalEq":>14}')
    print('-' * 90)
    for label, s in [
        ('Naissance',              portfolio_summary),
        ('Buy & Hold',         bh_summary),
        (f'ROC Momentum ({roc_period}-bar)',  roc_summary),
        ('S&P 500 (SPY)',      spy_summary),
    ]:
        if s is None:
            print(f'{label:<20} {"— no data —":>50}')
            continue
        print(f"{label:<20} {s['sharpe']:>8.3f} {s['max_drawdown']:>7.2f}% "
              f"{s['total_return_pct']:>9.2f}% {s['annualized_return_pct']:>7.2f}% "
              f"${s['final_equity']:>12,.2f}")
    print('=' * 90)

    _print_diagnostics(result['diagnostics'], n_days=len(result['equity_curve']))

    print(
        f"Average cash held: "
        f"{result['avg_cash_pct'] * 100:.1f}%"
    )

    print("\nPOSITION SIZES")
    print("=" * 80)

    for t in result["portfolio_trades"]:
        if "size" in t:
            print(
                f"{t['entry_date']} "
                f"{t['exit_date']} "
                f"Kelly={t['kelly_f']*100:6.1f}% "
                f"Actual={t['size']*100:6.1f}%"
            )

    plot_equity_and_drawdown(result['equity_curve'], bh_curve, roc_curve, spy_curve, roc_period)


if __name__ == '__main__':
    # --tickers=A,B,C or --universe (full universe.py list) override TICKERS
    # before dispatch, so main()/run_sensitivity_sweep() and every helper
    # reading the module-level TICKERS global see the same list.
    #
    # For a big universe, try ~20-25 names first and sanity-check
    # Sharpe/MaxDD/CAGR/cash% before running --universe (100 tickers).
    for _arg in sys.argv:
        if _arg.startswith('--tickers='):
            TICKERS = [t.strip().upper() for t in _arg.split('=', 1)[1].split(',') if t.strip()]
            print(f'[TICKERS] override: {len(TICKERS)} tickers from --tickers')
            break
    else:
        if '--universe' in sys.argv:
            import universe as _universe
            TICKERS = _universe.TICKERS
            print(f'[TICKERS] override: full universe.py, {len(TICKERS)} tickers')

    if '--sweep' in sys.argv:
        run_sensitivity_sweep(TICKERS, enable_shorts='--shorts' in sys.argv)
    else:
        main()