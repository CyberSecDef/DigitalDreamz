"""Corpus loaders for the three injection streams.

day_residue   — today's conversation transcripts; pulls salient fragments
world_events  — RSS headlines + first paragraphs
latent        — slow substrate; random chunks from a directory
"""
import re
import random
import time
from pathlib import Path
from typing import Optional

import feedparser


# ---------- DAY RESIDUE ----------

# crude salience: questions, exclamations, sentences with rare-ish words,
# sentences with named entities (rough heuristic — Capitalized mid-sentence words).
_QUESTION = re.compile(r"[^.!?]*\?")
_EXCLAIM = re.compile(r"[^.!?]*!")
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
        return random.choice(self.fragments)


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


class WorldEvents:
    REFRESH_SECONDS = 1800  # 30 min

    def __init__(
        self,
        feeds: list[str],
        blocklist: Optional[set[str]] = None,
        filter_enabled: bool = False,
    ):
        self.feeds = feeds
        self.blocklist = blocklist or set()
        self.filter_enabled = filter_enabled
        self.fragments: list[str] = []
        self._last_fetch = 0.0
        self._word_re = re.compile(r"[A-Za-z']+")
        self._refresh()

    def _is_blocked(self, text: str) -> bool:
        if not self.filter_enabled or not self.blocklist:
            return False
        words = (w.lower() for w in self._word_re.findall(text))
        return any(w in self.blocklist for w in words)

    def _refresh(self):
        now = time.time()
        if now - self._last_fetch < self.REFRESH_SECONDS and self.fragments:
            return
        new_frags = []
        for url in self.feeds:
            try:
                d = feedparser.parse(url)
                for entry in d.entries[:25]:
                    title = (entry.get("title") or "").strip()
                    summary = re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
                    if title and not self._is_blocked(title):
                        new_frags.append(title)
                    if summary and 30 < len(summary) < 280:
                        first = re.split(r"(?<=[.!?])\s", summary)[0]
                        if not self._is_blocked(first):
                            new_frags.append(first)
            except Exception:
                continue
        if new_frags:
            self.fragments = new_frags
            self._last_fetch = now

    def sample(self) -> Optional[str]:
        self._refresh()
        if not self.fragments:
            return None
        return random.choice(self.fragments)


# ---------- LATENT SUBSTRATE ----------

class LatentCorpus:
    def __init__(self, path: str, chunk_chars: int = 280):
        self.path = Path(path)
        self.chunk_chars = chunk_chars
        self.texts: list[str] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for f in self.path.glob("**/*"):
            if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
                try:
                    self.texts.append(f.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass

    def sample(self) -> Optional[str]:
        if not self.texts:
            return None
        text = random.choice(self.texts)
        if len(text) <= self.chunk_chars:
            return text.strip()
        start = random.randint(0, len(text) - self.chunk_chars)
        # try to start at a word boundary
        chunk = text[start:start + self.chunk_chars]
        # snap to whitespace edges
        first_space = chunk.find(" ")
        last_space = chunk.rfind(" ")
        if first_space > 0 and last_space > first_space:
            chunk = chunk[first_space:last_space]
        return chunk.strip()


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
            we["feeds"], blocklist=blocklist, filter_enabled=filter_enabled
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
