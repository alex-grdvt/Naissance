# Naissance

Naissance is a regime-adaptive quantitative equity trading system written in Python. It models market regimes with a Hidden Markov Model, scores each ticker with regime-weighted technical and factor-adjusted signals, sizes positions with Kelly-based and diversification rules, and records every decision in a SQLite database. It includes a walk-forward portfolio backtester with benchmark comparisons.

The project is a research and paper-trading system. It contains no broker integration and places no orders.

## Status

| Area | State |
|---|---|
| Ten-layer daily pipeline (`main.py`) | Implemented |
| Walk-forward portfolio backtest with baselines (`run_full_backtest.py`) | Implemented |
| `SCORE_GAP` sensitivity sweep (`--sweep`) | Implemented |
| Per-ticker sweep (`ticker_sweep.py`) | Implemented |
| Trade diagnostics (`diagnose_trades.py`) | Implemented |
| S&P 500 screen (`screen_universe.py`) | Work in progress: scraping, resumable CSV output and ranking are written; the backtest call is a placeholder |
| Validation on real data | Ongoing |

## Limitations

- Backtests are in-sample, run on a hand-selected universe of currently listed tickers, and therefore carry survivorship bias. They are not evidence of live performance.
- Transaction and short-borrow costs are set to zero by default (`ROUND_TRIP_COST_BPS`, `SHORT_BORROW_COST_BPS` in `layers/backtest_utils.py`).
- The Buy & Hold, ROC momentum and SPY baselines also apply no costs.
- There are no automated tests in this repository yet.

## How it works

The pipeline in `main.py` runs ten layers in order:

| Layer | Module | Output |
|---|---|---|
| L1 | `layers/kill_switch.py` | Macro stance: `long`, `neutral`, `short` or `cash`, from the yield curve, VIX and recession probability |
| L2 | `layers/regime_detection.py` | Per-ticker `bull`, `bear` or `sideways` label from a three-state Gaussian HMM, plus an independent high-volatility flag |
| L3 | `layers/alpha_score.py` | Long and short scores from `layers/signal_engine.py`, discounted by a Fama-French five-factor plus momentum regression (`layers/factor_decomposition.py`) |
| L4 | `layers/backtest_engine.py` | Single-ticker backtest that populates trade history for Kelly inputs |
| L5 | `layers/correlation_monitor.py` | Flagged correlated pairs and average portfolio correlation |
| L6 | `layers/decision_matrix.py` | `BUY`, `SHORT`, `EXIT`, `HOLD`, `BLOCK` or `CASH` per ticker |
| L7 | `asset_sizing.py`, `layers/position_sizing.py` | Position sizes: all-in for a single candidate, a diversified allocation for several |
| L8 | `layers/stop_loss.py` | Regime-aware trailing and range-based stops |
| L9 | `layers/replacement_logic.py` | Swaps stale holdings for stronger candidates |
| L10 | `portfolio_state.py` | Paper execution and persistence |

Design points:

- **Regimes.** In the backtest the HMM is refitted walk-forward on past data only, so no regime label uses future information. The portfolio backtest can also require a regime transition to agree with recent news sentiment (FinBERT) before it is accepted.
- **Indicators.** Signals use the rate of change of RSI and related momentum indicators as well as their levels.
- **Single source of truth.** Entry and exit rules live only in `layers/trade_logic.py`; the live pipeline and both backtests call it.
- **Sizing.** `asset_sizing.py` dispatches on the number of actionable tickers. The multi-ticker path ranks candidates by expected edge and applies per-ticker, category and asset-class caps with water-fill redistribution.

## Repository layout

```
main.py                 daily pipeline (L1-L10)
run_full_backtest.py    portfolio backtest driver
asset_sizing.py         L7 sizing (single and multi-ticker)
portfolio_state.py      SQLite schema and helpers
universe.py             tradable universe, categories and asset classes
diagnose_trades.py      trade-level diagnostics
ticker_sweep.py         per-ticker backtest sweep
screen_universe.py      S&P 500 screen (work in progress)
data/                   prices, macro, factor and sentiment fetchers
layers/                 the pipeline layers and shared trade logic
backtest/               simulation, baselines, reporting, calibration
```

## Setup

Python 3.11 was used for development.

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here      # free key from https://fred.stlouisfed.org
```

The SQLite database (`naissance.db`) is created next to the code on first run. The first sentiment run downloads the FinBERT model.

## Usage

```bash
python main.py                                       # daily pipeline
python run_full_backtest.py                          # default universe
python run_full_backtest.py --tickers=AAPL,NVDA      # chosen tickers
python run_full_backtest.py --universe               # every ticker in universe.py
python run_full_backtest.py --shorts                 # enable short entries
python run_full_backtest.py --fast                   # skip the sentiment fetch
python run_full_backtest.py --sweep                  # SCORE_GAP sensitivity sweep
python run_full_backtest.py --interval=5m --period=60d
python ticker_sweep.py                               # one backtest per ticker, ranked by Sharpe
python diagnose_trades.py                            # diagnostics for stored backtest trades
```

`run_full_backtest.py` reports Naissance alongside Buy & Hold, 12-month ROC momentum and SPY baselines and plots equity and drawdown.

## Data sources

Prices and news headlines come from Yahoo Finance (`yfinance`) and Google News (`gnews`), macro series from FRED, and factor returns from the Kenneth French data library. Headlines are scored with ProsusAI/FinBERT.