"""
multi_asset.py — L7 sizing for two or more actionable tickers (see
single_asset.py for the single-ticker case). Sits between L6
(decision_matrix) and execution.

Pipeline: compute_expected_edge (score_gap x regime x momentum x quality)
-> raw weights proportional to edge -> per-ticker/category/asset-class cap
with water-fill redistribution -> drawdown/vol throttle -> cash floor.
Kelly still bounds each name's risk-of-ruin; this layer only decides the
split.
"""

import numpy as np
import pandas as pd

from layers.position_sizing import (
    compute_kelly, get_trade_stats, get_portfolio_drawdown, get_consecutive_wins,
    DRAWDOWN_MILD, DRAWDOWN_SEVERE, DRAWDOWN_MILD_REDUCE, DRAWDOWN_SEVERE_REDUCE,
    CONSECUTIVE_WINS, CASH_FLOOR, ALL_IN, compute_all_in_weights,
)
from portfolio_state import get_maturity
import universe

# Diversification limits

MAX_TICKER_ALLOC   = 0.20   # per-ticker ceiling
MAX_CATEGORY_ALLOC = 0.40   # per-category ceiling
MAX_ASSET_CLASS_ALLOC = 0.65  # per-asset-class ceiling; keeps two correlated
                               # categories from filling the book together
TARGET_DEPLOYED  = 0.95   # >=5% left as a cash buffer, on top of CASH_FLOOR
WATERFILL_PASSES = 8      # redistribution rounds for the cap tiers to converge

# Category/asset-class membership comes from universe.py. A ticker missing
# from universe.py raises in _group_constraints rather than going uncapped.


def _group_constraints(tickers):
    """Ordered diversification tiers, finest-grained first. Raises if any
    ticker is not registered in universe.py."""
    unknown = [t for t in tickers if t not in universe.CATEGORY_OF]
    if unknown:
        raise ValueError(
            f'portfolio_optimizer: ticker(s) not found in universe.py, cannot '
            f'apply diversification caps: {unknown}. Add them to a category '
            f'in universe.py before trading them.'
        )
    return [
        (universe.CATEGORY_OF,    MAX_CATEGORY_ALLOC),
        (universe.ASSET_CLASS_OF, MAX_ASSET_CLASS_ALLOC),
    ]

# Expected edge

# Avg forward return by score_gap bucket, from the entry_diagnostics.csv
# sweep. Interpolated (see _expected_return_for_gap) rather than stepped, so
# gap=29 and gap=30 do not get a discontinuous allocation.
#
# Caveat: these buckets already have regime/quality baked into the sample
# they were measured from, so REGIME_MULTIPLIER and quality_mult below
# double-count that to some extent. Kept deliberately small (+-15% and
# 0.5-2.0x) so they tilt rather than dominate; edge is a ranking, not a
# rigorous forecast.
RETURN_TIERS = [
    (30, 0.0715),
    (25, 0.0578),
    (20, 0.0446),
    (10, 0.0434),
    (0,  0.0200),   # floor for a gap too small to have its own bucket
]

REGIME_MULTIPLIER = {'bull': 1.15, 'sideways': 1.00, 'bear': 0.85}

MOMENTUM_TILT_CAP = 0.10   # rsi_roc is a tiebreaker, not an independent edge source

# quality_mult = win_rate x avg_win/avg_loss (same source and MIN_TRADES=30
# floor as Kelly sizing). NEUTRAL_QUALITY matches position_sizing's below-
# MIN_TRADES default (p=0.50, b=1.20) so an untracked ticker gets mult=1.0.
# QUALITY_MULT_CLIP stops one exceptional track record from swamping every
# other factor.
NEUTRAL_QUALITY = 0.50 * 1.20        # = 0.60
QUALITY_MULT_CLIP = (0.5, 2.0)


def _expected_return_for_gap(score_gap):
    """Piecewise-linear interpolation between RETURN_TIERS' anchors, clamped
    flat beyond the lowest/highest gap."""
    anchors = sorted(RETURN_TIERS, key=lambda tr: tr[0])   # ascending by gap
    gaps = [g for g, _ in anchors]
    rets = [r for _, r in anchors]

    if score_gap <= gaps[0]:
        return rets[0]
    if score_gap >= gaps[-1]:
        return rets[-1]

    return float(np.interp(score_gap, gaps, rets))


def _ticker_quality_mult(ticker):
    """win_rate x avg_win/avg_loss, normalised against neutral and clipped.
    Uses the same maturity-gated stats source as _ticker_kelly_cap."""
    maturity = get_maturity(ticker)
    kelly_mode = maturity['kelly_mode']

    if kelly_mode == 'live':
        stats = get_trade_stats(ticker, source_filter='live')
    elif kelly_mode == 'backtest':
        stats = get_trade_stats(ticker, source_filter='backtest')
    else:
        stats = None

    if stats is None:
        return 1.0   # no trusted track record yet -- neutral

    p, b, _n = stats
    quality = p * b
    mult = quality / NEUTRAL_QUALITY
    return float(np.clip(mult, *QUALITY_MULT_CLIP))


def compute_expected_edge(candidate, ticker=None):
    """Expected return for a candidate, scaled by regime/momentum/quality.

    candidate: dict with direction, long_score, short_score, regime, rsi_roc.
    ticker: optional; folds in that ticker's quality multiplier. Omit for
    the score_gap/regime/momentum-only edge (e.g. the standalone demo below).

    Returns (edge, gap): edge is non-negative and comparable across
    tickers; gap is the raw score_gap, kept for diagnostics.
    """
    long_s  = candidate.get('long_score', 0) or 0
    short_s = candidate.get('short_score', 0) or 0
    gap     = (long_s - short_s) if candidate.get('direction') == 'long' else (short_s - long_s)
    gap     = max(0.0, gap)

    expected_return = _expected_return_for_gap(gap)
    regime_mult = REGIME_MULTIPLIER.get(candidate.get('regime'), 1.0)

    rsi_roc = candidate.get('rsi_roc')
    if rsi_roc is None:
        momentum_mult = 1.0
    else:
        tilt = np.clip(rsi_roc / 100.0, -MOMENTUM_TILT_CAP, MOMENTUM_TILT_CAP)
        momentum_mult = 1.0 + tilt

    quality_mult = _ticker_quality_mult(ticker) if ticker is not None else 1.0

    edge = expected_return * regime_mult * momentum_mult * quality_mult
    return max(0.0, edge), gap


def _ticker_kelly_cap(ticker):
    """Per-ticker Kelly ceiling on the optimizer's proposed weight, using the
    same live/backtest/default stats precedence as run_position_sizing."""
    maturity = get_maturity(ticker)
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


# Allocation

def allocate_portfolio(candidates, group_constraints=None):
    """
    candidates: {ticker: {direction, long_score, short_score, regime, rsi_roc}}
    group_constraints: overrides the default [(CATEGORY_OF, MAX_CATEGORY_ALLOC),
                        (ASSET_CLASS_OF, MAX_ASSET_CLASS_ALLOC)] tiers if given.

    Returns (weights, diagnostics):
      weights:      {ticker: fraction of portfolio_value}, sums to <= TARGET_DEPLOYED
      diagnostics:  {ticker: {edge, gap, kelly_cap, category, asset_class, capped_by}}
                    plus '_cash_residual' = fraction left undeployed
    """
    if not candidates:
        return {}, {'_cash_residual': TARGET_DEPLOYED}

    if len(candidates) == 1:
        raise ValueError(
            'multi_asset.allocate_portfolio called with a single candidate '
            f'({list(candidates)[0]!r}) — MAX_TICKER_ALLOC would cap it to '
            f'{MAX_TICKER_ALLOC:.0%} regardless of its Kelly cap, which is '
            'the wrong behaviour for a lone trade. Use '
            'single_asset.run_single_asset() instead when only one ticker '
            'is actionable.'
        )

    group_constraints = group_constraints or _group_constraints(list(candidates))

    edges = {}
    gaps = {}
    for t, c in candidates.items():
        edge, gap = compute_expected_edge(c, ticker=t)
        edges[t] = edge
        gaps[t] = gap

    total_edge = sum(edges.values())
    if total_edge <= 0:
        return {}, {'_cash_residual': TARGET_DEPLOYED}

    # step 2: raw proportional weights
    weights = {t: (edges[t] / total_edge) * TARGET_DEPLOYED for t in candidates}

    # per-ticker kelly caps, computed once
    kelly_caps = {t: _ticker_kelly_cap(t) for t in candidates}

    capped_by = {t: None for t in candidates}
    locked = {t: False for t in candidates}   # True once a ticker has hit any cap

    for _pass in range(WATERFILL_PASSES):
        changed = False

        # per-ticker caps (hard cap + kelly cap, whichever binds first)
        for t in candidates:
            if locked[t]:
                continue
            ceiling = min(MAX_TICKER_ALLOC, kelly_caps[t])
            if weights[t] > ceiling + 1e-9:
                weights[t] = ceiling
                locked[t] = True
                capped_by[t] = 'kelly' if kelly_caps[t] < MAX_TICKER_ALLOC else 'ticker_cap'
                changed = True

        # group caps, finest tier first (category, then asset class)
        # A member trimmed to its group's cap is locked, not just capped,
        # so the redistribution step below cannot push it back over.
        for group_map, max_alloc in group_constraints:
            groups = {}
            for t in candidates:
                g = group_map[t]
                groups.setdefault(g, []).append(t)

            for g, members in groups.items():
                group_total = sum(weights[t] for t in members)
                if group_total <= max_alloc + 1e-9:
                    continue
                unlocked = [t for t in members if not locked[t]]
                locked_total = sum(weights[t] for t in members if locked[t])
                room = max(0.0, max_alloc - locked_total)
                unlocked_total = sum(weights[t] for t in unlocked)
                if unlocked_total <= 0:
                    continue  # over cap purely from locked names -- nothing left to trim
                scale = room / unlocked_total if unlocked_total > 0 else 0.0
                for t in unlocked:
                    new_w = weights[t] * scale
                    if abs(new_w - weights[t]) > 1e-9:
                        weights[t] = new_w
                        changed = True
                    locked[t] = True
                    capped_by[t] = capped_by[t] or 'group_cap'

        # redistribute freed capital to unlocked names, proportional to edge
        deployed = sum(weights.values())
        residual = TARGET_DEPLOYED - deployed
        unlocked = [t for t in candidates if not locked[t]]
        if residual > 1e-6 and unlocked:
            unlocked_edge = sum(edges[t] for t in unlocked)
            if unlocked_edge > 0:
                for t in unlocked:
                    weights[t] += residual * (edges[t] / unlocked_edge)
                changed = True

        if not changed:
            break

    cash_residual = TARGET_DEPLOYED - sum(weights.values())

    diagnostics = {
        t: {
            'edge': round(edges[t], 4),
            'gap': round(gaps[t], 2),
            'kelly_cap': round(kelly_caps[t], 4),
            'category': universe.CATEGORY_OF.get(t, t),
            'asset_class': universe.ASSET_CLASS_OF.get(t, 'unknown'),
            'capped_by': capped_by[t],
        }
        for t in candidates
    }
    diagnostics['_cash_residual'] = round(max(0.0, cash_residual), 4)

    return {t: round(w, 4) for t, w in weights.items() if w > 1e-6}, diagnostics


# Risk throttle (reused from position_sizing, applied AFTER the split)

def _drawdown_reduction():
    """Sizing multiplier from portfolio drawdown, unless a recent win streak clears it."""
    drawdown = get_portfolio_drawdown()
    consecutive_wins = get_consecutive_wins()

    if drawdown > DRAWDOWN_SEVERE:
        return 0.0 if consecutive_wins >= CONSECUTIVE_WINS else DRAWDOWN_SEVERE_REDUCE
    if drawdown > DRAWDOWN_MILD:
        return 0.0 if consecutive_wins >= CONSECUTIVE_WINS else DRAWDOWN_MILD_REDUCE
    return 0.0


def run_multi_asset(tickers, scores, regimes, decisions, portfolio_value,
                             price_df=None, group_constraints=None, run_date=None):
    """
    Drop-in replacement for run_position_sizing at main.py's L7 call site.

    tickers:   the actionable list (decision in {BUY, SHORT})
    scores:    {ticker: breakdown}  as returned by alpha_score.score_all_tickers
                (needs 'raw_score'/'final_score' as long_score and 'short_score')
    regimes:   {ticker: regime}
    decisions: {ticker: {'action': ..., ...}}  to read direction (BUY->long, SHORT->short)
    run_date:  the date this allocation is for, as 'YYYY-MM-DD'. A backtest
               calling this per simulated day must pass that date, or every
               row gets stamped with today's wall-clock date instead.
    Returns sizes: {ticker: fraction_of_portfolio_value}, matching
    run_position_sizing's shape/units.

    Also writes one portfolio_allocations row per ticker (see
    portfolio_state.log_allocation) — the allocation decision, independent
    of what execution (L10) does with it.
    """
    from layers.regime_detection import is_high_vol_now
    from portfolio_state import log_allocation
    from datetime import datetime

    candidates = {}
    for t in tickers:
        bd = scores.get(t, {})
        direction = 'long' if decisions.get(t, {}).get('action') == 'BUY' else 'short'
        candidates[t] = {
            'direction':   direction,
            'long_score':  bd.get('final_score', bd.get('raw_score', 0)),
            'short_score': bd.get('short_score', 0),
            'regime':      regimes.get(t, 'sideways'),
            'rsi_roc':     bd.get('rsi_roc'),   # optional; None if not present upstream
        }

    if ALL_IN:
        # Bypass Kelly/category caps and quality multipliers; every
        # actionable ticker wants 100%, rationed by compute_all_in_weights.
        margins = {}
        for t, c in candidates.items():
            gap = (c['long_score'] - c['short_score']) if c['direction'] == 'long' \
                  else (c['short_score'] - c['long_score'])
            margins[t] = max(gap, 0.0)
        weights = compute_all_in_weights(margins, available_cash=portfolio_value,
                                          total_equity=portfolio_value)
        diagnostics = {
            t: {'edge': margins[t], 'gap': round(margins[t], 2), 'kelly_cap': None,
                'category': universe.CATEGORY_OF.get(t, t),
                'asset_class': universe.ASSET_CLASS_OF.get(t, 'unknown'),
                'capped_by': None}
            for t in candidates
        }
        diagnostics['_cash_residual'] = round(max(0.0, 1.0 - sum(weights.values())), 4)
    else:
        weights, diagnostics = allocate_portfolio(candidates, group_constraints=group_constraints)

    reduction = _drawdown_reduction() if not ALL_IN else 0.0
    if reduction:
        print(f'  [OPT] portfolio drawdown throttle active: -{reduction:.0%} across all sizes')

    run_date = run_date or datetime.today().strftime('%Y-%m-%d')
    cash_residual = diagnostics.get('_cash_residual', 0)

    sizes = {}
    for t, w in weights.items():
        high_vol = False
        if price_df is not None:
            try:
                high_vol = is_high_vol_now(price_df, t)
            except Exception:
                high_vol = False

        final_w = w * (1 - reduction)
        if high_vol:
            final_w *= 0.50
            print(f'  [OPT] {t} — high realized vol, halving size')

        d = diagnostics[t]
        print(f'  [OPT] {t} | edge={d["edge"]:.3f} gap={d["gap"]:.1f} '
              f'cat={d["category"]}/{d["asset_class"]} '
              f'| weight={w:.1%}{" (capped: " + d["capped_by"] + ")" if d["capped_by"] else ""} '
              f'-> final={final_w:.1%}')

        sizes[t] = round(final_w, 4)

        try:
            log_allocation(
                run_date=run_date, ticker=t, weight=sizes[t], raw_weight=w,
                edge=d['edge'], score_gap=d['gap'], regime=candidates[t]['regime'],
                category=d['category'], asset_class=d['asset_class'],
                capped_by=d['capped_by'], cash_residual=cash_residual,
            )
        except Exception as e:
            # logging failure should not take down the actual allocation
            print(f'  [OPT] {t} — allocation log write failed (non-fatal): {e}')

    print(f'  [OPT] cash residual: {cash_residual:.1%}')
    return sizes


if __name__ == '__main__':
    # Synthetic smoke test, no DB needed. Includes three correlated
    # categories (TECH/GROWTH/SEMIS) to exercise the asset-class tier.
    demo_candidates = {
        'TSLA': {'direction': 'long', 'long_score': 79.08, 'short_score': 26.18, 'regime': 'bull', 'rsi_roc': 4.1},
        'NVDA': {'direction': 'long', 'long_score': 69.67, 'short_score': 23.11, 'regime': 'sideways', 'rsi_roc': -0.2},
        'AAPL': {'direction': 'long', 'long_score': 68.99, 'short_score': 30.98, 'regime': 'bull', 'rsi_roc': 23.4},
        'PLTR': {'direction': 'long', 'long_score': 66.10, 'short_score': 28.40, 'regime': 'bull', 'rsi_roc': 15.2},
        'AMD':  {'direction': 'long', 'long_score': 60.20, 'short_score': 27.90, 'regime': 'sideways', 'rsi_roc': 6.8},
        'XOM':  {'direction': 'long', 'long_score': 45.02, 'short_score': 26.07, 'regime': 'bull', 'rsi_roc': -12.3},
        'GLD':  {'direction': 'long', 'long_score': 59.30, 'short_score': 25.34, 'regime': 'sideways', 'rsi_roc': -14.7},
        'TLT':  {'direction': 'long', 'long_score': 50.10, 'short_score': 24.00, 'regime': 'sideways', 'rsi_roc': 1.1},
    }

    # fake the kelly-cap/quality-mult DB lookups so this runs standalone
    globals()['_ticker_kelly_cap'] = lambda t: 0.9
    _demo_quality = {
        'NVDA': 0.63 * 3.34,   # -> clipped to QUALITY_MULT_CLIP max (2.0)
        'TSLA': 0.33 * 1.73,   # -> below neutral, mult < 1.0
    }
    globals()['_ticker_quality_mult'] = lambda t: float(
        np.clip(_demo_quality.get(t, NEUTRAL_QUALITY) / NEUTRAL_QUALITY, *QUALITY_MULT_CLIP)
    )

    weights, diag = allocate_portfolio(demo_candidates)
    print(f'{"ticker":6} {"weight":>8} {"edge":>8} {"gap":>7} {"category":>16} {"asset_class":>14}  capped_by')
    for t, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        d = diag[t]
        print(f'{t:6} {w:8.1%} {d["edge"]:8.3f} {d["gap"]:7.1f} {d["category"]:>16} {d["asset_class"]:>14}  {d["capped_by"] or "-"}')
    print(f'\ncash residual: {diag["_cash_residual"]:.1%}')
    print(f'total deployed: {sum(weights.values()):.1%}')

    asset_class_totals = {}
    category_totals = {}
    for t, w in weights.items():
        ac = universe.ASSET_CLASS_OF[t]
        cat = universe.CATEGORY_OF[t]
        asset_class_totals[ac] = asset_class_totals.get(ac, 0) + w
        category_totals[cat] = category_totals.get(cat, 0) + w
    print('\ncategory totals:')
    for c, w in category_totals.items():
        print(f'  {c:16} {w:.1%}')
    print('\nasset class totals:')
    for ac, w in asset_class_totals.items():
        print(f'  {ac:16} {w:.1%}')