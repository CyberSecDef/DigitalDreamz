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
CLAUDE_CODE_PROVIDERS = {"claude_code", "claude-code"}


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
    # Snapshot the real process environment so it can win over both files;
    # the overlay has to load with override=True to beat .env, which would
    # otherwise clobber one-off `MODEL_NAME=x python -m dreamer.main` runs.
    process_env = dict(os.environ)
    load_dotenv(root / ".env", override=False)
    env_name = os.environ.get("ENVIRONMENT", "dev")
    overlay = root / f".env.{env_name}"
    if overlay.exists():
        load_dotenv(overlay, override=True)
        os.environ.update(process_env)

    g = os.environ.__getitem__  # raise KeyError for required vars

    default_window = int(g("SAMPLING_CONTEXT_WINDOW_TOKENS"))
    window_by_phase = {
        phase: _opt_int(f"SAMPLING_CONTEXT_WINDOW_{phase.upper()}", default_window)
        for phase in PHASES
    }

    provider = g("MODEL_PROVIDER").strip()
    mode = os.environ.get("MODEL_MODE", "instruct").strip().lower()
    if provider in CLAUDE_CODE_PROVIDERS and mode == "base":
        raise ValueError(
            "MODEL_PROVIDER=claude_code has no base mode; set MODEL_MODE=instruct "
            "or use an ollama base model"
        )
    if (
        provider in CLAUDE_CODE_PROVIDERS
        and _bool(os.environ.get("CLAUDE_CODE_BARE", "false"))
        and not os.environ.get("ANTHROPIC_API_KEY", "").strip()
    ):
        raise ValueError(
            "CLAUDE_CODE_BARE=true authenticates only with ANTHROPIC_API_KEY; set it in "
            ".env.<environment> or the shell, or set CLAUDE_CODE_BARE=false to use the "
            "logged-in account (which leaks account context into the dream)"
        )
    # Auxiliary calls (self-state summaries, session distillation) are
    # instructions, so they need an instruct model even when the dreamer is a
    # base model. Defaults: Haiku via Claude Code when dreaming through Claude
    # Code, otherwise the dreaming model itself.
    if provider in CLAUDE_CODE_PROVIDERS:
        aux_default_provider, aux_default_name = provider, "claude-haiku-4-5-20251001"
    else:
        aux_default_provider, aux_default_name = provider, g("MODEL_NAME")
    aux_provider = os.environ.get("AUX_MODEL_PROVIDER", "").strip() or aux_default_provider
    aux_name = os.environ.get("AUX_MODEL_NAME", "").strip() or aux_default_name

    return {
        "environment": env_name,
        "session": {
            "duration_minutes": int(g("SESSION_DURATION_MINUTES")),
            "cycle_minutes": int(g("SESSION_CYCLE_MINUTES")),
            "perspective": g("SESSION_PERSPECTIVE"),
        },
        "model": {
            "provider": provider,
            "name": g("MODEL_NAME"),
            "mode": mode,
        },
        "aux_model": {
            "provider": aux_provider,
            "name": aux_name,
            "mode": "instruct",
        },
        "sampling": {
            "base_temp": float(g("SAMPLING_BASE_TEMP")),
            "temp_min": float(g("SAMPLING_TEMP_MIN")),
            "temp_max": float(g("SAMPLING_TEMP_MAX")),
            "top_p": float(g("SAMPLING_TOP_P")),
            "max_tokens_per_step": int(g("SAMPLING_MAX_TOKENS_PER_STEP")),
            "context_window_tokens": default_window,
            "context_window_by_phase": window_by_phase,
            "rem_peak_fraction": float(os.environ.get("SAMPLING_REM_PEAK_FRACTION", "0.75")),
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
                "sanitize_fragments": _bool(
                    os.environ.get("WORLD_SANITIZE_FRAGMENTS", "true")
                ),
            },
            "latent": {
                "path": g("CORPUS_LATENT_PATH"),
                "chunk_chars": int(g("CORPUS_LATENT_CHUNK_CHARS")),
            },
        },
        "monitor": {
            "topical_blocklist_path": os.environ.get(
                "TOPICAL_BLOCKLIST_PATH", "./data/topical_blocklist.txt"
            ),
            "stickiness_enabled": _bool(os.environ.get("STICKINESS_ENABLED", "false")),
            "stickiness_threshold": float(os.environ.get("STICKINESS_THRESHOLD", "0.5")),
            "stickiness_patience": int(os.environ.get("STICKINESS_PATIENCE", "3")),
        },
        "self_state": {
            "enabled": _bool(os.environ.get("SELF_STATE_ENABLED", "false")),
        },
        "accretion": {
            "enabled": _bool(os.environ.get("ACCRETION_ENABLED", "false")),
            "fixations_max": int(os.environ.get("ACCRETION_FIXATIONS_MAX", "200")),
            "distillations_max": int(os.environ.get("ACCRETION_DISTILLATIONS_MAX", "100")),
            "phase_summaries_max": int(os.environ.get("ACCRETION_PHASE_SUMMARIES_MAX", "300")),
        },
        "logging": {
            "db_path": g("LOG_DB_PATH"),
        },
    }
