"""Sleep-cycle modeling: phase detection + temperature oscillation,
plus stall + register-drift detection and clean-truncation helper.

A "cycle" is one drift→light→deep→rem→surface arc, ~15 min in config.
Within each cycle, position runs 0.0 → 1.0; phases and temperatures derive from it.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional


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


# ---------- phase-conditional context window ----------

_DEFAULT_WINDOW_BY_PHASE = {
    "drift": 800,
    "light": 1000,
    "deep": 1400,
    "rem": 600,
    "surface": 1000,
}


def window_for_phase(phase: str, by_phase: Optional[dict] = None, default: int = 1000) -> int:
    """Return the prompt-window token budget for a given phase.

    Falls back to the default if a phase isn't in the mapping.
    """
    table = by_phase if by_phase is not None else _DEFAULT_WINDOW_BY_PHASE
    return int(table.get(phase, default))


# ---------- register-drift (assistant/chat-mode contamination) ----------

# Phrases that signal the dream-state has collapsed back into a chat-assistant
# register. These are matched case-insensitively as substrings.
_DRIFT_PHRASES = (
    "I see what you",
    "Let me",
    "I can't provide",
    "I cannot provide",
    "I'll attempt",
    "I'll continue",
    "Here's a",
    "Here is a",
    "Let's",
    "I notice",
    "generative substrate",  # the model parroting its own system prompt
)

# Tighter second-person regex: only contractions and possessive (the spec's
# explicit list). Bare "you" appears too often in non-assistant prose to be
# useful as a trigger.
_DRIFT_REGEXES = [
    (re.compile(r"\byou(?:'ve|'re|'ll|'d|r|rself)\b", re.IGNORECASE), "second-person"),
    (re.compile(r"\*\*[^*\n]+\*\*"), "markdown-bold"),
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), "numbered-list"),
    (re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE), "markdown-heading"),
]

# Bracketed injections are explicitly "residue surfacing" per the system
# prompt — patterns inside them are not the model's register. Strip them
# from text before drift detection.
_BRACKETED = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def detect_register_drift(text: str, window_chars: int = 300) -> Optional[tuple[str, str]]:
    """Scan the tail of `text` for assistant-register markers.

    Returns (matched_pattern, snippet) on the first hit, or None.
    Bracketed injection fragments are stripped before scanning.
    """
    tail = text[-window_chars:] if len(text) > window_chars else text
    if not tail:
        return None
    scrubbed = _BRACKETED.sub(" ", tail)
    if not scrubbed.strip():
        return None
    lower = scrubbed.lower()
    for phrase in _DRIFT_PHRASES:
        if phrase.lower() in lower:
            return phrase, tail
    for rx, label in _DRIFT_REGEXES:
        if rx.search(scrubbed):
            return label, tail
    return None


# ---------- clean-sentence truncation (used by recovery surgery) ----------

# Sentence boundary characters appropriate to the dream register (drop ! and ?
# — those skew toward assistant/exclamation patterns).
_CLEAN_BOUNDARIES = (".", "…")


def truncate_to_clean_sentence(text: str, max_lookback: int = 500) -> tuple[str, int]:
    """Walk back from end of `text` up to `max_lookback` chars and truncate
    at the last boundary character that does NOT sit inside an assistant-
    register pattern.

    Returns (truncated_text, chars_removed). If no clean boundary is found
    within the lookback window, hard-cuts at the lookback edge.
    """
    if not text:
        return text, 0
    start = max(0, len(text) - max_lookback)
    region = text[start:]

    # Walk backward through sentence boundaries.
    for i in range(len(region) - 1, -1, -1):
        if region[i] in _CLEAN_BOUNDARIES:
            absolute = start + i + 1
            kept = text[:absolute]
            # Verify the kept tail doesn't itself end in a triggered region.
            tail = kept[-max_lookback:]
            if detect_register_drift(tail) is None:
                return kept, len(text) - absolute

    # No clean boundary found — hard cut at lookback edge.
    return text[:start], len(text) - start
