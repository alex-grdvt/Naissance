"""backtest — the portfolio backtest engine used by run_full_backtest.py.

  simulation.py  — universe loading and the walk-forward portfolio simulation
  baselines.py   — Buy & Hold, ROC momentum and SPY benchmark comparisons
  reporting.py   — summary statistics, diagnostics printout and equity/drawdown plots
  calibration.py — SCORE_GAP sensitivity sweep
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
