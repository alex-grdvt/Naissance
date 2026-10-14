"""backtest/reporting.py — portfolio and ticker summary statistics, the throttle
diagnostics printout, and the equity-curve and drawdown plots.
"""

import numpy as np
import pandas as pd

from layers.regime_detection import BEAR_SIZE_DAMPEN


# ── Portfolio-level summary stats (daily, mark-to-market) ────────────────────

def _portfolio_summary(equity_curve, periods_per_year_=252):
    """Summary statistics for the daily mark-to-market equity curve, or None if it is too short.

    periods_per_year_: bars per year at the curve's own frequency (252 for daily
    bars, more for intraday; see layers.backtest_utils.periods_per_year). Using the
    wrong factor misstates Sharpe materially: applying sqrt(252) to 5-minute bars
    overstates it by roughly sqrt(19656 / 252), about 8.8x.
    """
    if len(equity_curve) < 2:
        return None
    values = np.array([v for _, v in equity_curve])
    daily_returns = values[1:] / values[:-1] - 1

    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year_)) if daily_returns.std() > 0 else 0.0
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    max_dd = float(drawdowns.max()) * 100

    total_return_pct = (values[-1] / values[0] - 1) * 100
    first_date = pd.to_datetime(equity_curve[0][0])
    last_date  = pd.to_datetime(equity_curve[-1][0])
    years_elapsed = max((last_date - first_date).days / 365.25, 1 / 365.25)
    annualized_return_pct = ((values[-1] / values[0]) ** (1 / years_elapsed) - 1) * 100

    return {
        'sharpe':                round(sharpe, 4),
        'max_drawdown':          round(max_dd, 4),
        'final_equity':          round(float(values[-1]), 2),
        'total_return_pct':      round(total_return_pct, 4),
        'annualized_return_pct': round(annualized_return_pct, 4),
    }


def _ticker_summary(trades_for_ticker):
    """Per-ticker summary built from the notional actually deployed in the shared
    capital pool, not from a re-derived flat weight.
    """
    if not trades_for_ticker:
        return None
    n = len(trades_for_ticker)
    net_pnls = [t['net_pnl'] for t in trades_for_ticker]
    winners = [p for p in net_pnls if p > 0]
    losers  = [p for p in net_pnls if p <= 0]
    win_rate = len(winners) / n
    avg_win  = np.mean(winners) if winners else 0.0
    avg_loss = abs(np.mean(losers)) if losers else 0.0
    total_notional_pnl = sum(t['notional'] * (t['net_pnl'] / 100.0) for t in trades_for_ticker)
    return {
        'n_trades':  n,
        'win_rate':  round(win_rate, 4),
        'avg_win':   round(avg_win, 4),
        'avg_loss':  round(avg_loss, 4),
        'dollar_pnl': round(total_notional_pnl, 2),
    }


def _print_diagnostics(diag, n_days):
    """Prints how often each throttle fired, the kill-switch stance days and the exit-reason
    breakdown.
    """
    print(f'\n{"="*60}\nDIAGNOSTICS — did the throttles actually fire?\n{"="*60}')
    print(f"  correlation throttle — pair-flagged fires: {diag['corr_pair_fires']} | "
          f"avg-flagged fires: {diag['corr_avg_fires']}")
    print(f"  kill switch stance days (of {n_days}) — "
          f"neutral: {diag['kill_neutral_days']} | short: {diag['kill_short_days']} | "
          f"cash: {diag['kill_cash_days']}")
    print(f"  regime-confirmation blocks (sentiment disagreed, prior regime held): {diag['regime_confirm_blocks']}")
    print(f"  bear-regime size dampening fires (long entries taken at {BEAR_SIZE_DAMPEN}x size): "
          f"{diag['bear_dampen_fires']}")

    exit_counts = diag.get('exit_counts', {})
    total_exits = sum(exit_counts.values())
    if total_exits:
        print(f"\n  exit-reason breakdown ({total_exits} total closes):")
        for reason, n in exit_counts.most_common():
            print(f"    {reason:<20} {n:>5}  ({n/total_exits:.1%})")

    print(f"\n  entry-gate breakdown per ticker (top failure combos):")
    for ticker, counts in diag['gate_counts'].items():
        total = sum(counts.values())
        if total == 0:
            print(f"    {ticker}: never reached entry evaluation")
            continue
        print(f"    {ticker} — {total} entry attempts evaluated:")
        for combo, n in counts.most_common(5):
            label = '+'.join(combo)
            print(f"        {label:<45} {n:>5}  ({n/total:.1%})")

    if diag.get('enable_shorts'):
        print(f"\n  entry-gate breakdown per ticker — SHORT side (top failure combos):")
        for ticker, counts in diag['short_gate_counts'].items():
            total = sum(counts.values())
            if total == 0:
                print(f"    {ticker}: never reached short entry evaluation")
                continue
            print(f"    {ticker} — {total} short entry attempts evaluated:")
            for combo, n in counts.most_common(5):
                label = '+'.join(combo)
                print(f"        {label:<45} {n:>5}  ({n/total:.1%})")


def plot_equity_and_drawdown(equity_curve, bh_curve, roc_curve, spy_curve, roc_period):
    """Plots the Naissance equity curve against the baselines, with Naissance's own drawdown
    beneath.
    """
    import matplotlib.pyplot as plt

    # ===== EQUITY CURVE (Naissance vs. all baselines) =====

    eq = pd.DataFrame(
        equity_curve,
        columns=["date", "equity"],
    )
    eq["date"] = pd.to_datetime(eq["date"])

    # Same tuple-list format as equity_curve -- (date_str, value) -- so each
    # baseline gets its own DataFrame and plots on the same axes.
    baseline_curves = [
        ("Buy & Hold",                     bh_curve,  "tab:orange"),
        (f"ROC Momentum ({roc_period}-bar)", roc_curve, "tab:green"),
        ("S&P 500 (SPY)",                  spy_curve, "tab:red"),
    ]

    plt.figure(figsize=(14, 6))

    plt.plot(
        eq["date"],
        eq["equity"],
        linewidth=2,
        label="Naissance",
        color="tab:blue",
    )

    for label, curve, color in baseline_curves:
        if curve is None:
            continue
        b = pd.DataFrame(curve, columns=["date", "equity"])
        b["date"] = pd.to_datetime(b["date"])
        plt.plot(
            b["date"],
            b["equity"],
            linewidth=1.5,
            label=label,
            color=color,
            alpha=0.85,
        )

    plt.title("Naissance vs. Baselines — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ===== DRAWDOWN =====

    eq["peak"] = eq["equity"].cummax()

    eq["drawdown"] = (
        (eq["equity"] - eq["peak"])
        / eq["peak"]
    ) * 100

    plt.figure(figsize=(14, 6))

    plt.plot(
        eq["date"],
        eq["drawdown"],
    )

    plt.title("Naissance Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
