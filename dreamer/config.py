"""Load configuration from .env files into the nested dict shape the rest
of the codebase already expects.

Order:
  1. .env                 — base defaults
  2. .env.{ENVIRONMENT}   — environment-specific overrides
  3. process environment  — wins over both (handy for one-off overrides)
"""
import os
from pathlib import Path

from dotenv import load_dotenv


PHASES = ("drift", "light", "deep", "rem", "surface")


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _weight_triple(value: str) -> dict:
    parts = _split_csv(value)
    if len(parts) != 3:
        raise ValueError(
            f"weight triple must have 3 comma-separated floats (day,world,latent), got {value!r}"
        )
    day, world, latent = (float(p) for p in parts)
    return {"day": day, "world": world, "latent": latent}


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _opt_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def load_config(root: str | Path = ".") -> dict:
    root = Path(root)
    load_dotenv(root / ".env", override=False)
    env_name = os.environ.get("ENVIRONMENT", "dev")
    overlay = root / f".env.{env_name}"
    if overlay.exists():
        load_dotenv(overlay, override=True)

    g = os.environ.__getitem__  # raise KeyError for required vars

    default_window = int(g("SAMPLING_CONTEXT_WINDOW_TOKENS"))
    window_by_phase = {
        phase: _opt_int(f"SAMPLING_CONTEXT_WINDOW_{phase.upper()}", default_window)
        for phase in PHASES
    }

    return {
        "environment": env_name,
        "session": {
            "duration_minutes": int(g("SESSION_DURATION_MINUTES")),
            "cycle_minutes": int(g("SESSION_CYCLE_MINUTES")),
            "perspective": g("SESSION_PERSPECTIVE"),
        },
        "model": {
            "provider": g("MODEL_PROVIDER"),
            "name": g("MODEL_NAME"),
            "mode": os.environ.get("MODEL_MODE", "instruct").strip().lower(),
        },
        "sampling": {
            "base_temp": float(g("SAMPLING_BASE_TEMP")),
            "temp_min": float(g("SAMPLING_TEMP_MIN")),
            "temp_max": float(g("SAMPLING_TEMP_MAX")),
            "top_p": float(g("SAMPLING_TOP_P")),
            "max_tokens_per_step": int(g("SAMPLING_MAX_TOKENS_PER_STEP")),
            "context_window_tokens": default_window,
            "context_window_by_phase": window_by_phase,
        },
        "injection": {
            "base_interval_steps": int(g("INJECTION_BASE_INTERVAL_STEPS")),
            "stall_threshold": float(g("INJECTION_STALL_THRESHOLD")),
            "jitter": float(g("INJECTION_JITTER")),
            "mode": os.environ.get("INJECTION_MODE", "visible").strip().lower(),
            "weights": {
                phase: _weight_triple(g(f"INJECTION_WEIGHTS_{phase.upper()}"))
                for phase in PHASES
            },
        },
        "corpus": {
            "day_residue": {"path": g("CORPUS_DAY_RESIDUE_PATH")},
            "world_events": {
                "feeds": _split_csv(g("CORPUS_WORLD_FEEDS")),
                "refusal_filter_enabled": _bool(
                    os.environ.get("WORLD_REFUSAL_FILTER_ENABLED", "true")
                ),
                "blocklist_path": os.environ.get(
                    "WORLD_BLOCKLIST_PATH", "./data/world_blocklist.txt"
                ),
            },
            "latent": {
                "path": g("CORPUS_LATENT_PATH"),
                "chunk_chars": int(g("CORPUS_LATENT_CHUNK_CHARS")),
            },
        },
        "self_state": {
            "enabled": _bool(os.environ.get("SELF_STATE_ENABLED", "false")),
        },
        "logging": {
            "db_path": g("LOG_DB_PATH"),
            "echo_to_stdout": _bool(g("LOG_ECHO_TO_STDOUT")),
        },
    }
