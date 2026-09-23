"""Sleep-cycle modeling: phase detection + temperature oscillation,
plus stall + register-drift detection and clean-truncation helper.

A "cycle" is one drift→light→deep→rem→surface arc, ~15 min in config.
Within each cycle, position runs 0.0 → 1.0; phases and temperatures derive from it.
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


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


def temperature_for(
    pos: float,
    base: float,
    tmin: float,
    tmax: float,
    rem_peak_fraction: float = 0.75,
) -> float:
    """Piecewise sleep-cycle temperature curve.

    drift   (0.00–0.10): tmin → base, linear  (settling in)
    light   (0.10–0.30): base, with small jitter
    deep    (0.30–0.45): base → tmax, ramp    (max weirdness)
    rem     (0.45–0.85): plateau at base + rem_peak_fraction*(tmax-base),
                         with oscillation. 0.75 is the legacy value.
    surface (0.85–1.00): tmax → base, ramp down (waking)
    """
    if pos < 0.10:
        return _lerp(tmin, base, pos / 0.10)
    if pos < 0.30:
        return base + 0.08 * math.sin(pos * 40)
    if pos < 0.45:
        return _lerp(base, tmax, (pos - 0.30) / 0.15)
    if pos < 0.85:
        plateau = base + (tmax - base) * rem_peak_fraction
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
# register. Word-boundary anchored and case-insensitive: bare substring
# matching fired on "there's a" (→ "here's a") and "violet meadow"
# (→ "let me"), which cut clean dream prose. Label is what gets logged.
_DRIFT_PHRASES = [
    (re.compile(r"\bI see what you\b", re.IGNORECASE), "I see what you"),
    (re.compile(r"\blet me\b", re.IGNORECASE), "Let me"),
    (re.compile(r"\bI can(?:'t|not) provide\b", re.IGNORECASE), "I can't provide"),
    (re.compile(r"\bI'll (?:attempt|continue)\b", re.IGNORECASE), "I'll continue"),
    (re.compile(r"\bhere(?:'s| is) an?\b", re.IGNORECASE), "Here's a"),
    (re.compile(r"\blet's\b", re.IGNORECASE), "Let's"),
    (re.compile(r"\bI notice\b", re.IGNORECASE), "I notice"),
    # the model parroting its own system prompt
    (re.compile(r"\bgenerative substrate\b", re.IGNORECASE), "generative substrate"),
]

# Tighter second-person regex: only contractions and possessive (the spec's
# explicit list). Bare "you" appears too often in non-assistant prose to be
# useful as a trigger.
_DRIFT_REGEXES = [
    (re.compile(r"\byou(?:'ve|'re|'ll|'d|r|rself)\b", re.IGNORECASE), "second-person"),
    (re.compile(r"\*\*[^*\n]+\*\*"), "markdown-bold"),
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), "numbered-list"),
    (re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE), "markdown-heading"),
]

# Injection fragments are wrapped in ‹...› angle brackets by main.py — they
# are residue from the corpus, not the model's register. Strip them from text
# before any of the stickiness / drift / topical detectors run.
_INJECTION_RE = re.compile(r"\n*‹[^‹›]*›\n*")
# A slice can start or end partway through a fragment: text before the first
# › with no ‹ ahead of it is the tail of an injection, and a trailing ‹ with
# no closing › is the head of one.
_ORPHAN_HEAD_RE = re.compile(r"^[^‹]*›")
_ORPHAN_TAIL_RE = re.compile(r"‹[^›]*$")

RECEDING_CLOSE = "‹/receding›"


def scrub_injections(text: str, replacement: str = " ") -> str:
    """Remove every ‹…› fragment, including orphaned halves at the edges of a
    slice, leaving only the model's own text."""
    text = _ORPHAN_HEAD_RE.sub(replacement, text, count=1)
    text = _ORPHAN_TAIL_RE.sub(replacement, text, count=1)
    return _INJECTION_RE.sub(replacement, text)


def strip_brackets(text: str) -> str:
    """Drop ‹ › characters from generated text (summaries, distillations) so
    it can never produce nested brackets once re-wrapped as an injection."""
    return text.replace("‹", "").replace("›", "").strip()


def strip_markers(text: str) -> str:
    """Buffer text → model text only: scrub every injection and wrapper,
    then any stray brackets. Used before buffer text leaves the loop."""
    return strip_brackets(re.sub(r"[ \t]+", " ", scrub_injections(text)))


def _normalize_quotes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _model_tail(text: str, window_chars: int) -> str:
    """Last `window_chars` of model-authored text. Scrubs before slicing so a
    fragment straddling the window edge can't leak into the scan; the 3x
    over-read leaves room for the fragments that get removed."""
    region = text[-(window_chars * 3 + 600):]
    return _normalize_quotes(scrub_injections(region))[-window_chars:]


def detect_register_drift(text: str, window_chars: int = 300) -> Optional[tuple[str, str]]:
    """Scan the tail of `text` for assistant-register markers.

    Returns (matched_pattern, snippet) on the first hit, or None.
    Bracketed injection fragments are stripped before scanning.
    """
    tail = _model_tail(text, window_chars)
    if not tail.strip():
        return None
    for rx, label in _DRIFT_PHRASES:
        if rx.search(tail):
            return label, tail
    for rx, label in _DRIFT_REGEXES:
        if rx.search(tail):
            return label, tail
    return None


# ---------- step-start echo (instruct models repeating the last sentence) ----------

_TERMINAL = ".!?…\"'”’"


class EchoFilter:
    """Hold back the start of a step while it matches the buffer's last
    sentence. The buffer reaches instruct models as a user message (Claude
    5-series refuses assistant prefill), and they sometimes open by echoing
    the final sentence, then revise it:

        …more like a weather. | The recipient's own name, … more like a weather — something

    A full echo is dropped. If the model then extends the old sentence
    rather than ending it, `extends` is set and the caller should remove the
    buffer's final punctuation so the extension attaches to it. Sentences
    under `min_len` chars are left alone, since short refrains are often
    deliberate."""

    def __init__(self, prev_text: str, min_len: int = 40):
        tail = prev_text.rstrip()
        last = ""
        if tail and tail[-1] in _TERMINAL:
            last = re.split(r"(?<=[.!?…])\s+", tail)[-1]
        body = last.rstrip(_TERMINAL + " ")
        ok = len(body) >= min_len and "‹" not in body and "›" not in body
        self.body = body if ok else ""
        self.terminal = last[len(body):].strip() if ok else ""
        self.held = ""
        self.passing = not self.body
        self.extends = False
        self.echoed = False

    def feed(self, delta: str) -> str:
        if self.passing:
            return delta
        self.held += delta
        probe = self.held.lstrip()
        if len(probe) <= len(self.body):
            if self.body.startswith(probe):
                return ""  # still matching; keep holding
            return self._release()
        if not probe.startswith(self.body):
            return self._release()
        rest = probe[len(self.body):]
        self.passing = True
        self.echoed = True
        self.held = ""
        if self.terminal and rest.startswith(self.terminal):
            return rest[len(self.terminal):].lstrip()  # exact repeat: drop it
        if rest[:1] in _TERMINAL:
            return rest.lstrip(_TERMINAL).lstrip()  # repeat, different stop
        self.extends = True  # "weather — something": continues the old sentence
        return rest

    def finish(self) -> str:
        """Release anything still held when the stream ends."""
        if self.passing or self.held.lstrip() == self.body:
            self.held = ""
            return ""
        return self._release()

    def _release(self) -> str:
        out, self.held, self.passing = self.held, "", True
        return out


def trailing_terminal_len(text: str) -> int:
    """Length of the trailing whitespace + sentence punctuation run."""
    stripped = text.rstrip()
    stripped = stripped.rstrip(_TERMINAL)
    return len(text) - len(stripped)


# ---------- clean-sentence truncation (used by recovery surgery) ----------

# Sentence boundary characters appropriate to the dream register (drop ! and ?
# — those skew toward assistant/exclamation patterns).
_CLEAN_BOUNDARIES = (".", "…")


def truncate_to_clean_sentence(
    text: str,
    max_lookback: int = 500,
    is_clean: Optional[Callable[[str], bool]] = None,
) -> tuple[str, int]:
    """Walk back from end of `text` up to `max_lookback` chars and truncate
    at the last boundary character where the kept text passes `is_clean`
    and the boundary is not inside an injected ‹…› fragment.

    `is_clean` should run every detector that could have fired: checking
    only register drift let a date or email in an earlier, complete sentence
    survive the cut. Defaults to the register-drift check alone.

    Returns (truncated_text, chars_removed). If no clean boundary is found
    within the lookback window, hard-cuts at the lookback edge (backed off
    to the start of any fragment the edge would split).
    """
    if not text:
        return text, 0
    if is_clean is None:
        def is_clean(kept: str) -> bool:
            return detect_register_drift(kept) is None
    start = max(0, len(text) - max_lookback)
    region = text[start:]

    for i in range(len(region) - 1, -1, -1):
        if region[i] not in _CLEAN_BOUNDARIES:
            continue
        absolute = start + i + 1
        kept = text[:absolute]
        if kept.rfind("‹") > kept.rfind("›"):
            continue  # boundary is inside an injected fragment
        if is_clean(kept):
            return kept, len(text) - absolute

    cut = start
    open_pos = text.rfind("‹", 0, cut)
    if open_pos > text.rfind("›", 0, cut):
        cut = open_pos
    return text[:cut], len(text) - cut


# ---------- harness leak (Claude Code context bleeding into the dream) ----------

# Claude Code's logged-in mode attaches the account email, working directory,
# git status, model identity and date to every call. Under the dream prompt
# the model treats these as residue and writes them into the dream. Any
# trace of them is contamination: it gets cut by recovery surgery so it
# never reaches the transcript or the accreted corpus.
_HARNESS_LEAK_REGEXES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "email"),
    (re.compile(r"(?<![\w/])/(?:tmp|home|usr|var)\b"), "path"),
    (re.compile(r"\bgit (?:repo|repository|status)\b|\.git\b|\bworking director", re.IGNORECASE), "workspace"),
    (re.compile(r"\b(?:claude|sonnet|opus|haiku|fable)\b|\bmodel id\b|\bknowledge cutoff\b", re.IGNORECASE), "model-identity"),
    (re.compile(r"system[- ]reminder|\bplatform: linux\b|\bos version\b", re.IGNORECASE), "harness"),
]


def detect_harness_leak(
    text: str, window_chars: int = 300, extra_terms: tuple[str, ...] = ()
) -> Optional[tuple[str, str]]:
    """Scan the tail of `text` for harness context leaking into the dream.
    `extra_terms` are literal strings (e.g. the account's email local part)
    matched case-insensitively. Returns (label, snippet) or None."""
    tail = _model_tail(text, window_chars)
    if not tail.strip():
        return None
    for rx, label in _HARNESS_LEAK_REGEXES:
        if rx.search(tail):
            return label, tail
    lower = tail.lower()
    for term in extra_terms:
        if term and term.lower() in lower:
            return "account", tail
    return None


# ---------- date leak (the one line --bare can't strip) ----------

# Claude Code attaches "Today's date is YYYY-MM-DD." to every call, bare mode
# included, and the dream picks it up verbatim. A full ISO date is never
# dream material, so it is safe to cut; a bare year ("1911") is left alone.
_DATE_LEAK_RE = re.compile(
    r"\b(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    r"|\btoday's date is\b",
    re.IGNORECASE,
)


def redact_date_leaks(text: str) -> str:
    """Drop every sentence carrying an ISO date or the harness date line
    from generated text (summaries, distillations). The aux model receives
    the same date line as the dreamer and adds it on its own."""
    out_lines = []
    for line in text.splitlines():
        if not _DATE_LEAK_RE.search(line):
            out_lines.append(line)
            continue
        kept = [s for s in re.split(r"(?<=[.!?…])\s+", line) if not _DATE_LEAK_RE.search(s)]
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines).strip()


def detect_date_leak(text: str, window_chars: int = 300) -> Optional[tuple[str, str]]:
    """Scan the tail of `text` for an ISO date or the harness's date line.
    Returns ("iso-date", snippet) or None."""
    tail = _model_tail(text, window_chars)
    if _DATE_LEAK_RE.search(tail):
        return "iso-date", tail
    return None


# ---------- topical-drift (commentary-blog / culture-war basins) ----------

# Module-level cache so we only read each blocklist file once.
_topical_cache: dict[str, list[str]] = {}


def load_topical_patterns(path: str) -> list[str]:
    """Load patterns from a blocklist file, one pattern per line. Comment
    lines (starting with #) and blank lines are skipped. Patterns are kept
    in their original case but matched case-insensitively at scan time.

    Multi-word patterns are matched as substrings; single tokens use
    word-boundary matching.
    """
    if path in _topical_cache:
        return _topical_cache[path]
    p = Path(path) if not isinstance(path, Path) else path
    patterns: list[str] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            patterns.append(s)
    _topical_cache[path] = patterns
    return patterns


def detect_topical_drift(
    text: str, patterns: list[str], window_chars: int = 300
) -> Optional[tuple[str, str]]:
    """Scan the tail of `text` for any topical-blocklist pattern.

    Returns (matched_pattern, snippet) on the first hit, or None.
    Bracketed injection fragments are stripped before scanning, same as
    the register detector.
    """
    if not patterns:
        return None
    tail = _model_tail(text, window_chars)
    if not tail.strip():
        return None
    lower = tail.lower()
    for pat in patterns:
        pat_lower = pat.lower()
        if " " in pat:
            if pat_lower in lower:
                return pat, tail
        else:
            if re.search(rf"\b{re.escape(pat_lower)}\b", lower):
                return pat, tail
    return None


# ---------- register-stickiness (content-word recycling rate) ----------

# Common function words filtered so the metric measures *content* recycling,
# not function-word overlap (which any English prose has in abundance).
_STICKINESS_STOPWORDS = frozenset("""
the a an and or of in on at to with from for by as is are was were be been being
it its this that these those into onto upon over under like through which who
what when where why how each every all any some no not but yet still even so
very more most much many few less had has have having did do does done
""".split())

_CONTENT_WORD_RE = re.compile(r"[a-z]+")


def _content_words(text: str) -> list[str]:
    return [
        w for w in _CONTENT_WORD_RE.findall(text.lower())
        if len(w) >= 4 and w not in _STICKINESS_STOPWORDS
    ]


def register_stickiness(
    text: str,
    recent_chars: int = 500,
    history_chars: int = 3000,
) -> float:
    """Fraction of recent content words that are recycled from the preceding
    history window. 0.0 → recent text is all-new vocabulary; 1.0 → every
    content word repeats from history.

    Content words: alphabetic, length ≥ 4, after stopword filtering. Injection
    fragments (‹...›) are stripped from both windows before scoring so the
    metric reflects only the model's own vocabulary recycling.

    Only text after the most recent ‹/receding› marker is scored: the
    receding block is the history that recovery just pushed away, and
    scoring against it re-fires stickiness immediately after every recovery.

    Returns 0.0 when there's not enough text to score.
    """
    text = text.rsplit(RECEDING_CLOSE, 1)[-1]
    needed = recent_chars + history_chars
    model_text = scrub_injections(text[-(needed * 2):])
    if len(model_text) < needed:
        return 0.0
    recent_text = model_text[-recent_chars:]
    history_text = model_text[-needed:-recent_chars]
    recent_words = _content_words(recent_text)
    if not recent_words:
        return 0.0
    history_set = set(_content_words(history_text))
    recycled = sum(1 for w in recent_words if w in history_set)
    return recycled / len(recent_words)
