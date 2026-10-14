"""trade_logic.py — the single authoritative source for Naissance's entry and exit
decision rules (score-gap, RSI and fast-momentum gates) and the RSI indicators
they are built on.

backtest_engine.py, run_full_backtest.py (through backtest/simulation.py) and
the live decision matrix all call into this module instead of keeping their own
copies, so a rule changed here applies to every caller.

Do not duplicate these constants or conditions elsewhere. If a caller needs a
variant, extend the functions below with a parameter rather than forking the
logic.
"""

# ── Gate constants (canonical values — see module docstring) ──
SCORE_GAP = 5              # long score must beat short by this margin to enter, and vice versa
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
FAST_MOM_THRESH = -15               # RSI-ROC(3) floor; blocks long entries during a bull-to-bear transition
FAST_MOM_THRESH_SHORT = -FAST_MOM_THRESH  # mirror for short entries


# ── Shared indicators ──

def compute_rsi(closes, period=14):
    """Standard Wilder-style RSI over period bars."""
    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def compute_rsi_roc(closes, period=14, roc_period=3):
    """N-bar rate of change of RSI; fast-momentum guard that leads the HMM at regime transitions."""
    rsi = compute_rsi(closes, period=period)
    return rsi.diff(roc_period)


# ── Entry gates ──
# These are the only implementations of the long/short entry decision. Every
# simulation loop (single-ticker backtest, portfolio backtest) must call them
# rather than re-deriving the condition inline.

def long_entry_gates(long_score, short_score, rsi, rsi_roc, regime):
    """Whether a long entry is allowed right now.

    Returns (wants_long: bool, failed_gates: list[str]) — failed_gates is
    empty iff wants_long is True, and names which gate(s) blocked the entry
    otherwise (useful for diagnostics/gate-count logging).

    A bear regime does not block long entries outright, because a 3-state HMM
    can mislabel a state; it requires double the score gap instead.
    """
    rsi_gate_ok = (rsi < RSI_OVERBOUGHT) if regime != 'bull' else (rsi < 90)
    fast_mom_gate_ok = (rsi_roc > FAST_MOM_THRESH)
    score_gap_required = SCORE_GAP * 2 if regime == 'bear' else SCORE_GAP

    failed = []
    if not (long_score > short_score + score_gap_required):
        failed.append('score_gap')
    if not rsi_gate_ok:
        failed.append('rsi')
    if not fast_mom_gate_ok:
        failed.append('fast_mom')

    return (not failed), failed


def short_entry_gates(long_score, short_score, rsi, rsi_roc, regime, enable_shorts=True):
    """Whether a short entry is allowed right now.

    Returns (wants_short: bool, failed_gates: list[str]). Always
    (False, ['shorts_disabled']) when enable_shorts is False, so callers can
    unconditionally call this and branch on the result rather than guarding
    the call site.
    """
    if not enable_shorts:
        return False, ['shorts_disabled']

    failed = []
    if regime == 'bull':
        failed.append('regime_bull')

    rsi_gate_ok = (rsi > RSI_OVERSOLD) if regime != 'bear' else (rsi > 10)
    fast_mom_gate_ok = (rsi_roc < FAST_MOM_THRESH_SHORT)

    if not (short_score > long_score + SCORE_GAP):
        failed.append('score_gap')
    if not rsi_gate_ok:
        failed.append('rsi')
    if not fast_mom_gate_ok:
        failed.append('fast_mom')

    return (not failed), failed


# ── Regime-flip exit checks ──
# Kept in one place so that long and short exits share a single definition.

def long_exit_due_to_regime(regime):
    """True once regime has flipped bearish enough to force a long exit."""
    return regime == 'bear'


def short_exit_due_to_regime(regime):
    """True once regime has flipped bullish enough to force a short exit."""
    return regime == 'bull'