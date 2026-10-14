"""
single_asset.py — L7 sizing for a portfolio with exactly one actionable
ticker (see multi_asset.py for the two-or-more case).

multi_asset.py's MAX_TICKER_ALLOC / category / asset-class caps exist to
stop one name or sector from crowding out other actionable names in a
multi-position book. With exactly one actionable ticker there is nothing
else to crowd out, so those caps do not apply here — the only ceiling is
this ticker's own Kelly cap (bounded at KELLY_CAP). Shared account-level
controls (drawdown throttle, high-vol halving, cash floor) still apply in
both files, since those are account protection rather than diversification.
"""
import sys
# Aliases '__main__' to 'single_asset' in sys.modules so that when this file
# is run directly, run_full_backtest.py's later `from single_asset import
# run_single_asset` resolves to the already-loaded module instead of
# re-importing (and re-executing) this file as a second module object.
sys.modules.setdefault('single_asset', sys.modules['__main__'])

KELLY_FLOOR = 0.7

from layers.position_sizing import (
    compute_kelly, get_trade_stats, get_portfolio_drawdown, get_consecutive_wins,
    DRAWDOWN_MILD, DRAWDOWN_SEVERE, DRAWDOWN_MILD_REDUCE, DRAWDOWN_SEVERE_REDUCE,
    CONSECUTIVE_WINS, CASH_FLOOR, KELLY_CAP, ALL_IN,
)
from portfolio_state import get_maturity


def ticker_kelly_cap(ticker):
    """Per-ticker Kelly ceiling, same stats-source precedence as multi_asset.py."""
    maturity   = get_maturity(ticker)
    kelly_mode = maturity['kelly_mode']

    if kelly_mode == 'live':
        stats = get_trade_stats(ticker, source_filter='live')
    elif kelly_mode == 'backtest':
        stats = get_trade_stats(ticker, source_filter='backtest')
    else:
        stats = None

    if stats is None:
        p, b = 0.50, 1.20
    else:
        p, b, _n = stats

    return compute_kelly(p, b)


def size_single_asset(ticker, high_vol=False):
    """Fraction of portfolio_value for this ticker, with no other candidate
    competing for capital -- so no ticker/category/asset-class cap, only
    its own Kelly cap."""
    if ALL_IN:
        # Nothing else competing for capital, so ALL_IN just means 100%.
        # High-vol halving still applies (account protection, not sizing).
        fraction = 0.50 if high_vol else 1.0
        return max(0.0, round(fraction, 4))

    kelly_f = ticker_kelly_cap(ticker)

    drawdown         = get_portfolio_drawdown()
    consecutive_wins = get_consecutive_wins()

    if drawdown > DRAWDOWN_SEVERE:
        reduction = 0.0 if consecutive_wins >= CONSECUTIVE_WINS else DRAWDOWN_SEVERE_REDUCE
    elif drawdown > DRAWDOWN_MILD:
        reduction = 0.0 if consecutive_wins >= CONSECUTIVE_WINS else DRAWDOWN_MILD_REDUCE
    else:
        reduction = 0.0

    fraction = max(KELLY_FLOOR, min(kelly_f * (1 - reduction), 1))

    if high_vol:
        fraction *= 0.50

    return max(0.0, round(fraction, 4))


def run_single_asset(ticker, portfolio_value, price_df=None):
    """Drop-in for main.py's L7 call site when exactly one ticker is
    actionable. Returns {ticker: fraction}, same shape as run_multi_asset.
    CASH_FLOOR is enforced by the caller, not here."""
    high_vol = False
    if price_df is not None:
        from layers.regime_detection import is_high_vol_now
        try:
            high_vol = is_high_vol_now(price_df, ticker)
        except Exception:
            high_vol = False

    fraction = size_single_asset(ticker, high_vol=high_vol)
    return {ticker: fraction}


if __name__ == '__main__':
    # Standalone smoke test: exercises this file's own sizing path directly
    # against the real DB, without spinning up a full backtest.
    from portfolio_state import initialise_db
    initialise_db()

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    sizes = run_single_asset(ticker, portfolio_value=100_000)
    print(f'{ticker}: {sizes[ticker]:.1%} of portfolio_value')