"""backtest/calibration.py — SCORE_GAP sensitivity sweep.

Reruns the full portfolio simulation across a range of SCORE_GAP values (the
default of 10 plus looser candidates) to test whether the entry gate is too
tight or the model genuinely has low edge.

run_portfolio_backtest reads SCORE_GAP as a module-level name in
backtest.simulation, so the sweep sets simulation.SCORE_GAP for each candidate
and restores the original value afterwards. layers/trade_logic.py and the live
path are unaffected.

Sentiment is held neutral (sentiment_df=None) for every candidate so that the
sweep isolates the effect of SCORE_GAP.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

from portfolio_state import initialise_db
from data.factor_data import fetch_factor_data
from data.macro_data import fetch_macro_data
from layers.backtest_utils import BACKTEST_YEARS

from backtest import simulation
from backtest.reporting import _portfolio_summary


SCORE_GAP_SWEEP_VALUES = [10, 7, 5, 3, 0]   # 10 is the default gate


def run_sensitivity_sweep(tickers, sweep_values=None, enable_shorts=False):
    """Runs the portfolio simulation once per SCORE_GAP value and prints a comparison table.

    tickers is passed explicitly; sweep_values defaults to SCORE_GAP_SWEEP_VALUES.
    The database retains only the last swept value, so rerun run_full_backtest.py
    without --sweep afterwards to store the chosen threshold.
    """
    sweep_values = sweep_values or SCORE_GAP_SWEEP_VALUES
    original_gap = simulation.SCORE_GAP

    initialise_db()

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - relativedelta(years=BACKTEST_YEARS)).strftime('%Y-%m-%d')
    factor_df = fetch_factor_data(start, end)
    try:
        macro_df = fetch_macro_data(start, end)
    except Exception:
        macro_df = None

    print(f'\n{"="*90}\nSCORE_GAP SENSITIVITY SWEEP (sentiment held neutral to isolate the effect)\n{"="*90}')
    print(f'{"SCORE_GAP":>10} {"Trades":>8} {"PassRate":>10} {"Sharpe":>8} {"MaxDD":>8} {"CAGR":>8}')
    print('-' * 90)

    for gap in sweep_values:
        simulation.SCORE_GAP = gap
        # Clear earlier rows so trades from different thresholds do not accumulate.
        simulation._clear_previous(tickers)
        result  = simulation.run_portfolio_backtest(tickers, start, end, factor_df,
                                          sentiment_df=None, macro_df=macro_df,
                                          enable_shorts=enable_shorts)
        summary = _portfolio_summary(result['equity_curve'])

        gate_counts     = result['diagnostics']['gate_counts']
        total_attempts  = sum(sum(c.values()) for c in gate_counts.values())
        total_pass      = sum(c.get(('pass',), 0) for c in gate_counts.values())
        pass_rate       = (total_pass / total_attempts) if total_attempts else 0.0
        n_trades        = len(result['portfolio_trades'])

        if summary:
            print(f"{gap:>10} {n_trades:>8} {pass_rate:>9.1%} {summary['sharpe']:>8.3f} "
                  f"{summary['max_drawdown']:>7.2f}% {summary['annualized_return_pct']:>7.2f}%")
        else:
            print(f"{gap:>10} {n_trades:>8} {pass_rate:>9.1%} {'— no trades —':>30}")

    simulation.SCORE_GAP = original_gap
    print('=' * 90)
    print(f"NOTE: the database now reflects only the last swept value ({sweep_values[-1]}); "
          f"rerun run_full_backtest.py without --sweep to store the chosen threshold.")
