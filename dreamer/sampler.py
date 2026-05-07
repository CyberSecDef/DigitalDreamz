"""Sleep-cycle modeling: phase detection + temperature oscillation.

A "cycle" is one drift→light→deep→rem→surface arc, ~15 min in config.
Within each cycle, position runs 0.0 → 1.0; phases and temperatures derive from it.
"""
import math
from dataclasses import dataclass


PHASE_BOUNDARIES = [
    (0.10, "drift"),
    (0.30, "light"),
    (0.45, "deep"),
    (0.85, "rem"),
    (1.01, "surface"),
]


def cycle_position(elapsed_seconds: float, cycle_seconds: float) -> float:
    return (elapsed_seconds % cycle_seconds) / cycle_seconds


def phase_for(pos: float) -> str:
    for boundary, name in PHASE_BOUNDARIES:
        if pos < boundary:
            return name
    return "surface"


def temperature_for(pos: float, base: float, tmin: float, tmax: float) -> float:
    """Piecewise sleep-cycle temperature curve.

    drift   (0.00–0.10): tmin → base, linear  (settling in)
    light   (0.10–0.30): base, with small jitter
    deep    (0.30–0.45): base → tmax, ramp    (max weirdness)
    rem     (0.45–0.85): high plateau ~0.7*tmax with oscillation (vivid)
    surface (0.85–1.00): tmax → base, ramp down (waking)
    """
    if pos < 0.10:
        return _lerp(tmin, base, pos / 0.10)
    if pos < 0.30:
        return base + 0.08 * math.sin(pos * 40)
    if pos < 0.45:
        return _lerp(base, tmax, (pos - 0.30) / 0.15)
    if pos < 0.85:
        plateau = base + (tmax - base) * 0.75
        return plateau + 0.12 * math.sin(pos * 35)
    return _lerp(tmax, base, (pos - 0.85) / 0.15)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


# ---------- stall detection ----------

def stall_score(recent_text: str, window_chars: int = 600) -> float:
    """Crude n-gram overlap between two halves of the recent buffer.

    Returns 0.0 (no overlap, fresh) → 1.0 (heavy repetition, stalled).
    """
    if len(recent_text) < window_chars * 2:
        return 0.0
    a = recent_text[-window_chars * 2 : -window_chars]
    b = recent_text[-window_chars:]
    bg_a = _bigrams(a)
    bg_b = _bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    inter = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return inter / union if union else 0.0


def _bigrams(text: str) -> set:
    words = text.lower().split()
    return set(zip(words, words[1:])) if len(words) > 1 else set()


@dataclass
class PhaseState:
    """Tracks current phase to detect transitions for logging."""
    current: str = "drift"

    def update(self, pos: float) -> tuple[str, bool]:
        new = phase_for(pos)
        changed = new != self.current
        prev = self.current
        self.current = new
        return prev, changed
