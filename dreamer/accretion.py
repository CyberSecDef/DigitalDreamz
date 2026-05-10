"""Latent corpus self-extension. Three tiers, all under the latent path:

    fixations/   — online captures of the buffer when stickiness fires
    sessions/    — post-session distillations via a single complete_once call
    preserved/   — manual, hand-curated material (no auto-write)

All three are picked up automatically by LatentCorpus on the next session.
Weighting is governed by corpus/latent/weights.txt; default rules are
shipped there. Failures here are non-fatal — accretion never blocks the
dream loop.
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import llm


DISTILL_PROMPT = (
    "Read the dream transcript below. Extract the recurring images, "
    "vivid fragments, and unresolved questions that surfaced. Output as "
    "a list of short fragments — one per line, no numbering, no preamble, "
    "no explanation. Match the dream's register: spare, concrete, oneiric. "
    "Skip anything generic. Aim for 8–20 fragments."
)

_MAX_DISTILL_INPUT_CHARS = 12000


def ensure_dirs(latent_path: str) -> Path:
    root = Path(latent_path)
    (root / "fixations").mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "preserved").mkdir(parents=True, exist_ok=True)
    return root


def write_fixation(latent_path: str, session_id: int, step: int, snippet: str) -> Optional[Path]:
    """Tier A: capture the recent buffer when stickiness recovery fires.
    Filename carries metadata; file content is plain fragment so the
    chunk sampler doesn't pollute future injections with frontmatter."""
    snippet = snippet.strip()
    if not snippet:
        return None
    try:
        root = ensure_dirs(latent_path)
        target = root / "fixations" / f"{session_id:04d}-step{step:04d}.md"
        target.write_text(snippet + "\n", encoding="utf-8")
        return target
    except OSError as e:
        print(f"accretion: fixation write failed: {e}", file=sys.stderr)
        return None


def write_distillation(
    latent_path: str,
    session_id: int,
    transcript: str,
    model_cfg: dict,
) -> Optional[Path]:
    """Tier B: one-shot summary of the session transcript, written to
    sessions/. Returns the path written, or None on failure / empty input."""
    if not transcript.strip():
        return None
    tail = transcript[-_MAX_DISTILL_INPUT_CHARS:]
    try:
        text = llm.complete_once(
            provider=model_cfg["provider"],
            name=model_cfg["name"],
            mode="instruct",
            system_prompt=DISTILL_PROMPT,
            user_prompt=tail,
            temperature=0.6,
            top_p=0.9,
            max_tokens=600,
        )
    except Exception as e:
        print(f"accretion: distillation LLM call failed: {e}", file=sys.stderr)
        return None
    text = text.strip()
    if not text:
        return None
    try:
        root = ensure_dirs(latent_path)
        date = datetime.fromtimestamp(time.time()).strftime("%Y%m%d")
        target = root / "sessions" / f"{session_id:04d}-{date}.md"
        target.write_text(text + "\n", encoding="utf-8")
        return target
    except OSError as e:
        print(f"accretion: distillation write failed: {e}", file=sys.stderr)
        return None


def prune(latent_path: str, subdir: str, max_files: int) -> int:
    """FIFO cleanup by mtime. Keeps the newest `max_files`, deletes the rest.
    Returns the number of files removed. No-op if max_files <= 0."""
    if max_files <= 0:
        return 0
    folder = Path(latent_path) / subdir
    if not folder.exists():
        return 0
    files = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix in {".md", ".txt"}),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(files) - max_files
    if excess <= 0:
        return 0
    removed = 0
    for f in files[:excess]:
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed
