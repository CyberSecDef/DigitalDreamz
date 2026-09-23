import json
import stat

import pytest

from dreamer import llm


def _fake_claude(tmp_path, events, exit_code=0, sleep=0):
    lines = "\n".join(json.dumps(e) for e in events)
    script = tmp_path / "claude"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "sys.stdin.read()\n"
        f"print({lines!r}, flush=True)\n"
        f"time.sleep({sleep})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _delta(text):
    return {"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}}


RESULT = {"type": "result", "subtype": "success", "is_error": False,
          "total_cost_usd": 0.0025,
          "usage": {"input_tokens": 800, "cache_read_input_tokens": 50, "output_tokens": 40}}


def _stream(max_tokens=100, tracker=None):
    return llm.stream_completion(
        provider="claude_code", name="claude-sonnet-5", mode="instruct",
        system_prompt="sys", user_prompt="buffer", temperature=1.4, top_p=0.9,
        max_tokens=max_tokens, tracker=tracker)


def test_claude_code_streams_text_and_records_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", _fake_claude(
        tmp_path, [{"type": "system", "subtype": "init"}, _delta("Fog "), _delta("in the hall."), RESULT]))
    tracker = llm.UsageTracker()
    assert "".join(_stream(tracker=tracker)) == "Fog in the hall."
    assert tracker.prompt_total == 850 and tracker.completion_total == 40
    assert tracker.cost_total == pytest.approx(0.0025)


def test_claude_code_runs_on_to_sentence_end(tmp_path, monkeypatch):
    # 40-char budget lands mid-sentence; the step ends at the next full stop.
    deltas = ["The hallway outside ", "keeps forgetting to end. ", "It goes ", "and goes."]
    monkeypatch.setenv("CLAUDE_CODE_BIN", _fake_claude(
        tmp_path, [_delta(d) for d in deltas], sleep=30))
    tracker = llm.UsageTracker()
    out = "".join(_stream(max_tokens=10, tracker=tracker))
    assert out == "The hallway outside keeps forgetting to end. "
    assert tracker.had_approx


def test_claude_code_hard_cap_cuts_at_word_boundary(tmp_path, monkeypatch):
    # No sentence end before 1.5x budget (60 chars): stop at a word edge.
    words = [_delta("drift ")] * 20
    monkeypatch.setenv("CLAUDE_CODE_BIN", _fake_claude(tmp_path, words, sleep=30))
    out = "".join(_stream(max_tokens=10))
    assert out.strip() and len(out) <= 60
    assert all(w == "drift" for w in out.split())


@pytest.mark.parametrize("text,emitted,expected", [
    ("abc", 0, ("abc", False)),                           # under budget
    ("end of it. More", 5, ("end of it. ", True)),         # sentence end past budget
    ("ok.” Then", 9, ("ok.” ", True)),                     # closing quote kept
    ("line\nnext", 8, ("line\n", True)),                   # newline is a boundary
    ("to the end.", 8, ("to the end. ", True)),            # sentence end at delta edge
    ("so it goes and goes", 8, ("so it goes and goes", False)),  # still under cap
])
def test_clip_at_budget(text, emitted, expected):
    assert llm._clip_at_budget(text, emitted, budget=10, hard_cap=30) == expected


def test_claude_code_error_result_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", _fake_claude(
        tmp_path, [{"type": "result", "subtype": "error_during_execution", "is_error": True,
                    "result": "rate limited"}], exit_code=1))
    with pytest.raises(RuntimeError, match="rate limited"):
        list(_stream())


def test_claude_code_crash_without_result_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", _fake_claude(tmp_path, [_delta("half")], exit_code=1))
    with pytest.raises(RuntimeError, match="without a result"):
        list(_stream())


def test_claude_code_command_isolates_the_call(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_BARE", raising=False)
    cmd = llm.claude_code_command("claude-sonnet-5", "SYS")
    for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"


def test_claude_code_rejects_base_mode():
    with pytest.raises(ValueError):
        list(llm.stream_completion("claude_code", "claude-sonnet-5", "base", "", "x", 1.0, 1.0, 10))


def test_anthropic_via_litellm_clamps_temperature(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    list(llm.stream_completion("anthropic", "claude-sonnet-5", "instruct", "s", "u", 1.7, 0.97, 10))
    assert seen["temperature"] == 1.0 and "top_p" not in seen


def test_bare_swaps_safe_mode(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BARE", "true")
    cmd = llm.claude_code_command("claude-sonnet-5", "SYS")
    assert "--bare" in cmd and "--safe-mode" not in cmd
    assert llm.claude_code_leak_terms() == ()
