import pytest

from dreamer.config import load_config

BASE = """ENVIRONMENT=dev
SESSION_DURATION_MINUTES=30
SESSION_CYCLE_MINUTES=15
SESSION_PERSPECTIVE=third
MODEL_PROVIDER=claude_code
MODEL_NAME=claude-sonnet-5
SAMPLING_BASE_TEMP=1.0
SAMPLING_TEMP_MIN=0.8
SAMPLING_TEMP_MAX=1.3
SAMPLING_TOP_P=0.97
SAMPLING_MAX_TOKENS_PER_STEP=80
SAMPLING_CONTEXT_WINDOW_TOKENS=1000
INJECTION_BASE_INTERVAL_STEPS=5
INJECTION_STALL_THRESHOLD=0.3
INJECTION_JITTER=0.4
INJECTION_WEIGHTS_DRIFT=0.2,0.3,0.5
INJECTION_WEIGHTS_LIGHT=0.2,0.3,0.5
INJECTION_WEIGHTS_DEEP=0.2,0.3,0.5
INJECTION_WEIGHTS_REM=0.2,0.3,0.5
INJECTION_WEIGHTS_SURFACE=0.2,0.3,0.5
CORPUS_DAY_RESIDUE_PATH=./today
CORPUS_WORLD_FEEDS=
CORPUS_LATENT_PATH=./latent
CORPUS_LATENT_CHUNK_CHARS=280
LOG_DB_PATH=./x.db
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    for line in BASE.splitlines():
        monkeypatch.delenv(line.split("=")[0], raising=False)
    for key in ("MODEL_MODE", "AUX_MODEL_PROVIDER", "AUX_MODEL_NAME",
                "CLAUDE_CODE_BARE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(BASE)
    (tmp_path / ".env.dev").write_text("MODEL_NAME=from-overlay\nSESSION_CYCLE_MINUTES=30\n")
    return tmp_path


def test_process_env_beats_overlay(root, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "from-shell")
    cfg = load_config(root)
    assert cfg["model"]["name"] == "from-shell"
    assert cfg["session"]["cycle_minutes"] == 30  # overlay still beats .env


def test_claude_code_aux_defaults_to_haiku(root):
    cfg = load_config(root)
    assert cfg["aux_model"] == {
        "provider": "claude_code", "name": "claude-haiku-4-5-20251001", "mode": "instruct"}


def test_claude_code_base_mode_rejected(root, monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "base")
    with pytest.raises(ValueError):
        load_config(root)


def test_bare_requires_api_key(root, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BARE", "true")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        load_config(root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert load_config(root)["model"]["provider"] == "claude_code"
