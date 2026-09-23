"""Corpus loaders for the three injection streams.

day_residue   — today's conversation transcripts; pulls salient fragments
world_events  — RSS headlines + first paragraphs
latent        — slow substrate; random chunks from a directory
"""
import fnmatch
import re
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import feedparser


def _clean_fragment(text: str) -> str:
    """Drop ‹ › so a fragment can't nest inside the ‹…› wrapper main.py adds
    (nested brackets defeat the injection scrubber)."""
    return text.replace("‹", "").replace("›", "").strip()


# ---------- DAY RESIDUE ----------

# crude salience: questions, exclamations, sentences with named entities
# (rough heuristic — Capitalized mid-sentence words).
_SENT = re.compile(r"[^.!?\n]+[.!?]")


def _is_salient(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 20 or len(s) > 280:
        return False
    if "?" in s or "!" in s:
        return True
    # mid-sentence capitalization (proper noun heuristic)
    words = s.split()
    if len(words) > 4:
        mid_caps = sum(1 for w in words[1:] if w[:1].isupper() and w[:1].isalpha())
        if mid_caps >= 2:
            return True
    return False


class DayResidue:
    def __init__(self, path: str):
        self.path = Path(path)
        self.fragments: list[str] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for f in self.path.glob("*"):
            if f.suffix.lower() not in {".txt", ".md", ".log"}:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for sent in _SENT.findall(text):
                if _is_salient(sent):
                    self.fragments.append(sent.strip())

    def sample(self) -> Optional[str]:
        if not self.fragments:
            return None
        return _clean_fragment(random.choice(self.fragments))


# ---------- WORLD EVENTS ----------

def _load_blocklist(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    tokens: set[str] = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.add(line.lower())
    return tokens


# Strip URLs and metadata-label scaffolding that gives the model a strong
# news-article / HN-post genre signal.
_URL_RE = re.compile(r"https?://\S+")
_METADATA_LABEL_RE = re.compile(
    r"\b(?:Article URL|Comments URL|Comments|# Comments|Points?|Posted by|"
    r"Source|Read more|Continue reading)\s*[:\-]\s*",
    re.IGNORECASE,
)
# Common dateline/wire patterns: "BEIJING:", "REUTERS —", "NEW YORK (AP)",
# "(Reuters) —". The third alt catches a CITY followed by a parenthesized
# wire-service tag without a separator.
_DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*\s*[:—–\-])"
    r"|^\s*\(?(?:Reuters|AP|AFP|Bloomberg|Press Association|PA)\)?\s*[:—–\-]?"
    r"|^\s*[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*\s*\([A-Za-z]+\)\s*",
    re.MULTILINE,
)


def _sanitize_world_fragment(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _METADATA_LABEL_RE.sub("", text)
    text = _DATELINE_RE.sub("", text)
    # Collapse repeated whitespace introduced by the substitutions.
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WorldEvents:
    REFRESH_SECONDS = 1800  # 30 min
    FETCH_TIMEOUT = 10      # per feed; feedparser alone never times out

    def __init__(
        self,
        feeds: list[str],
        blocklist: Optional[set[str]] = None,
        filter_enabled: bool = False,
        sanitize: bool = True,
    ):
        self.feeds = feeds
        self.blocklist = blocklist or set()
        self.filter_enabled = filter_enabled
        self.sanitize = sanitize
        self.fragments: list[str] = []
        self._last_fetch = 0.0
        self._refreshing = threading.Lock()
        self._word_re = re.compile(r"[A-Za-z']+")
        # First fetch is synchronous (bounded by FETCH_TIMEOUT per feed) so
        # the session starts with world material; later refreshes run in the
        # background so the dream loop never blocks on the network.
        self._refresh()

    def _is_blocked(self, text: str) -> bool:
        if not self.filter_enabled or not self.blocklist:
            return False
        words = (w.lower() for w in self._word_re.findall(text))
        return any(w in self.blocklist for w in words)

    def _maybe_sanitize(self, text: str) -> str:
        return _sanitize_world_fragment(text) if self.sanitize else text

    def _fetch(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "dreamer/1.0"})
        with urllib.request.urlopen(req, timeout=self.FETCH_TIMEOUT) as resp:
            return feedparser.parse(resp.read())

    def _refresh(self):
        if not self._refreshing.acquire(blocking=False):
            return
        try:
            # Stamp the attempt up front: a failed refresh waits the full
            # interval instead of retrying on every sample.
            self._last_fetch = time.time()
            new_frags = []
            for url in self.feeds:
                try:
                    d = self._fetch(url)
                except Exception:
                    continue
                for entry in d.entries[:25]:
                    title = self._maybe_sanitize(
                        (entry.get("title") or "").strip()
                    )
                    summary = self._maybe_sanitize(
                        re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
                    )
                    if title and not self._is_blocked(title):
                        new_frags.append(title)
                    if summary and 30 < len(summary) < 280:
                        first = re.split(r"(?<=[.!?])\s", summary)[0]
                        if not self._is_blocked(first):
                            new_frags.append(first)
            if new_frags:
                self.fragments = new_frags
        finally:
            self._refreshing.release()

    def _maybe_refresh_async(self):
        if time.time() - self._last_fetch < self.REFRESH_SECONDS:
            return
        threading.Thread(target=self._refresh, daemon=True).start()

    def sample(self) -> Optional[str]:
        self._maybe_refresh_async()
        if not self.fragments:
            return None
        return _clean_fragment(random.choice(self.fragments))


# ---------- LATENT SUBSTRATE ----------

class LatentCorpus:
    """Loads .txt/.md files from the latent path. Files are weighted, with
    per-file or per-subdirectory overrides read from `weights.txt` files
    placed alongside the content.

    `weights.txt` format (one entry per line):
        <pattern> <weight>
    Pattern is matched against `Path.relative_to(latent_root)` as a
    fnmatch-style glob (e.g. `essays/*`, `poetry/*.txt`, `red_book.md`).
    Patterns are evaluated in order; the first match wins. Files that match
    no pattern get weight 1.0. A weight of 0 excludes the file from sampling
    without removing it from disk.
    """

    def __init__(self, path: str, chunk_chars: int = 280):
        self.path = Path(path)
        self.chunk_chars = chunk_chars
        # One entry per file: the file's paragraphs (or lines, for one-
        # fragment-per-line files such as session distillations).
        self.texts: list[list[str]] = []
        self.weights: list[float] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        weight_rules = self._collect_weight_rules()
        for f in self.path.glob("**/*"):
            if f.is_file() and f.suffix.lower() in {".txt", ".md"} and f.name != "weights.txt":
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                rel = f.relative_to(self.path).as_posix()
                w = self._weight_for(rel, weight_rules)
                if w <= 0:
                    continue
                units = self._units(content)
                if not units:
                    continue
                self.texts.append(units)
                self.weights.append(w)

    def _units(self, content: str) -> list[str]:
        """Split a file into self-contained sampling units. Paragraphs split
        on blank lines; an oversized paragraph made of several lines is split
        on lines instead, so list-shaped files yield one fragment per line."""
        units: list[str] = []
        for para in re.split(r"\n\s*\n", content):
            para = para.strip()
            if not para:
                continue
            if len(para) > self.chunk_chars and "\n" in para:
                units.extend(l.strip() for l in para.splitlines() if l.strip())
            else:
                units.append(" ".join(para.split()))
        return units

    def _collect_weight_rules(self) -> list[tuple[str, float]]:
        """Read every weights.txt file under the latent path. Rules from
        deeper files override shallower ones for matches under that subtree."""
        rules: list[tuple[str, float]] = []
        for wf in sorted(self.path.glob("**/weights.txt")):
            base = wf.parent.relative_to(self.path).as_posix()
            if base == ".":
                base = ""
            try:
                lines = wf.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                # Strip inline comments (anything from a #-preceded-by-
                # whitespace token onward).
                hash_pos = s.find(" #")
                if hash_pos >= 0:
                    s = s[:hash_pos].rstrip()
                if not s:
                    continue
                parts = s.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                pattern, raw = parts
                try:
                    weight = float(raw)
                except ValueError:
                    continue
                full_pattern = f"{base}/{pattern}" if base else pattern
                rules.append((full_pattern, weight))
        return rules

    def _weight_for(self, rel_path: str, rules: list[tuple[str, float]]) -> float:
        for pattern, weight in rules:
            if fnmatch.fnmatchcase(rel_path, pattern):
                return weight
        return 1.0

    def sample(self) -> Optional[str]:
        """Pick a file by weight, then a paragraph. Paragraphs within the
        chunk budget are returned whole; longer ones yield a run of whole
        sentences starting at a random sentence, so a fragment never opens
        or closes mid-sentence."""
        if not self.texts:
            return None
        units = random.choices(self.texts, weights=self.weights, k=1)[0]
        unit = random.choice(units)
        if len(unit) <= self.chunk_chars:
            return _clean_fragment(unit)
        sentences = re.split(r"(?<=[.!?…])\s+", unit)
        i = random.randrange(len(sentences))
        chunk = sentences[i]
        for nxt in sentences[i + 1:]:
            if len(chunk) + 1 + len(nxt) > self.chunk_chars:
                break
            chunk = f"{chunk} {nxt}"
        if len(chunk) > self.chunk_chars:
            # a single sentence longer than the budget: cut at a word edge
            chunk = chunk[: self.chunk_chars].rsplit(" ", 1)[0] + "…"
        return _clean_fragment(chunk)


# ---------- COMPOSITE ----------

class Corpus:
    def __init__(self, config: dict):
        c = config["corpus"]
        self.day = DayResidue(c["day_residue"]["path"])

        # Refusal filter is auto-disabled in base mode (no chat-tuned refusals
        # to dodge), regardless of the env-flag value.
        we = c["world_events"]
        is_instruct = config.get("model", {}).get("mode", "instruct") == "instruct"
        filter_enabled = bool(we.get("refusal_filter_enabled", False)) and is_instruct
        blocklist = _load_blocklist(we.get("blocklist_path", "")) if filter_enabled else set()
        self.world = WorldEvents(
            we["feeds"],
            blocklist=blocklist,
            filter_enabled=filter_enabled,
            sanitize=bool(we.get("sanitize_fragments", True)),
        )

        self.latent = LatentCorpus(
            c["latent"]["path"], c["latent"].get("chunk_chars", 280)
        )
        self.weights = config["injection"]["weights"]

    def sample_latent(self) -> Optional[str]:
        """Used by recovery surgery — high-dissociation source."""
        return self.latent.sample()

    def sample_for_phase(self, phase: str) -> tuple[str, str]:
        """Returns (source_name, fragment). May return ('', '') if all empty."""
        w = self.weights.get(phase, {"day": 0.4, "world": 0.4, "latent": 0.2})
        sources = []
        weights = []
        if self.day.fragments:
            sources.append(("day", self.day))
            weights.append(w.get("day", 0.0))
        if self.world.fragments:
            sources.append(("world", self.world))
            weights.append(w.get("world", 0.0))
        if self.latent.texts:
            sources.append(("latent", self.latent))
            weights.append(w.get("latent", 0.0))
        if not sources or sum(weights) == 0:
            return "", ""
        name, src = random.choices(sources, weights=weights, k=1)[0]
        frag = src.sample()
        return name, (frag or "")
