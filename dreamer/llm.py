"""Multi-provider streaming: Claude Code (default) or litellm.

Provider `claude_code` shells out to the `claude` CLI in print mode. Each
call is stripped down: --safe-mode (no hooks, CLAUDE.md, plugins, MCP), no
tools, no thinking, no session persistence, and our own system prompt in
place of Claude Code's. The CLI exposes no sampling controls, so temperature
and top_p are ignored on this path and max_tokens is enforced client-side by
cutting the stream (see prompts.phase_hint for the substitute).

Logged-in (subscription) auth still attaches context the flags can't remove:
the account email, working directory, git status, model identity, and date.
The dream prompt reads anything unexplained as residue, so the model weaves
it into the dream; sampler.detect_harness_leak catches it for recovery.
CLAUDE_CODE_BARE=true passes --bare, which removes that context entirely but
authenticates only with ANTHROPIC_API_KEY.

Every other provider routes through litellm, with two prompting modes:
- 'instruct': chat-template path (system + user messages). For chat/instruct
  models. Uses litellm.completion.
- 'base': raw text-completion path (no chat template, buffer sent as prefix).
  For non-chat-tuned base models. Uses litellm.text_completion. For Ollama,
  this routes through /api/generate rather than /api/chat.
"""
import json
import os
import re
import subprocess
import tempfile
import threading
from typing import Iterator, Optional
import litellm

from .config import CLAUDE_CODE_PROVIDERS

# Providers that accept temperature/top_p. Anthropic's API caps temperature
# at 1.0 and recent Claude models reject temperature and top_p together.
_ANTHROPIC_TEMP_MAX = 1.0

if os.environ.get("LITELLM_DEBUG", "").lower() in {"1", "true", "yes"}:
    litellm._turn_on_debug()


class UsageTracker:
    """Cumulative + delta token counters, with $ estimate via litellm.model_cost.

    Provider usage is preferred; when a stream returns no usage object (some
    self-hosted endpoints), the caller falls back to a char/4 approximation
    via `add_approx`. The `approx` flag on each delta records whether the
    numbers were measured or estimated."""

    def __init__(self):
        self.prompt_total = 0
        self.completion_total = 0
        self.prompt_delta = 0
        self.completion_delta = 0
        self.cost_total = 0.0
        self.cost_delta = 0.0
        self.had_approx = False  # true if any window contained estimated counts

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        approx: bool = False,
        cost: Optional[float] = None,
    ) -> None:
        self.prompt_total += prompt_tokens
        self.completion_total += completion_tokens
        self.prompt_delta += prompt_tokens
        self.completion_delta += completion_tokens
        if cost is None:
            cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        self.cost_total += cost
        self.cost_delta += cost
        if approx:
            self.had_approx = True

    def add_approx(self, prompt_text: str, completion_text: str, model: str) -> None:
        # Rough heuristic: ~4 chars per token for English. Good enough for a
        # cost gauge when the provider didn't return real usage.
        p = max(1, len(prompt_text) // 4)
        c = max(0, len(completion_text) // 4)
        self.add(p, c, model, approx=True)

    def snapshot_and_reset_delta(self) -> dict:
        snap = {
            "prompt_total": self.prompt_total,
            "completion_total": self.completion_total,
            "prompt_delta": self.prompt_delta,
            "completion_delta": self.completion_delta,
            "cost_total": self.cost_total,
            "cost_delta": self.cost_delta,
            "approx": self.had_approx,
        }
        self.prompt_delta = 0
        self.completion_delta = 0
        self.cost_delta = 0.0
        self.had_approx = False
        return snap


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Look up per-token rates from litellm.model_cost. Returns 0.0 if the
    model isn't in the table (e.g. local Ollama models, custom endpoints)."""
    try:
        cost_map = getattr(litellm, "model_cost", {}) or {}
    except Exception:
        return 0.0
    # Try the model string as-is, then strip a provider prefix (e.g. "openai/").
    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
    for key in candidates:
        entry = cost_map.get(key)
        if not entry:
            continue
        p_rate = entry.get("input_cost_per_token", 0.0) or 0.0
        c_rate = entry.get("output_cost_per_token", 0.0) or 0.0
        return prompt_tokens * p_rate + completion_tokens * c_rate
    return 0.0


def _model_string(provider: str, name: str, mode: str) -> str:
    # In base mode, force the plain `ollama/` prefix (the chat endpoint can't
    # serve raw completions); for other providers, leave as-is.
    if mode == "base" and provider == "ollama_chat":
        provider = "ollama"
    if provider in {"anthropic", "openai"}:
        return name
    return f"{provider}/{name}"


def _extract_delta(chunk) -> Optional[str]:
    """Pull the text delta out of a streaming chunk regardless of API shape."""
    try:
        choice = chunk.choices[0]
    except (AttributeError, IndexError):
        return None
    # chat-completion shape
    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if content:
            return content
    # text-completion shape
    text = getattr(choice, "text", None)
    if text:
        return text
    return None


def _extract_usage(chunk) -> Optional[tuple[int, int]]:
    """Return (prompt_tokens, completion_tokens) from a streaming chunk if
    present, else None. Providers attach usage to the final chunk when
    stream_options={'include_usage': True} is honored."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    p = getattr(usage, "prompt_tokens", None)
    c = getattr(usage, "completion_tokens", None)
    if p is None and c is None:
        # Some providers nest as dict
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens")
            c = usage.get("completion_tokens")
    if p is None and c is None:
        return None
    return int(p or 0), int(c or 0)


def is_claude_code(provider: str) -> bool:
    return provider in CLAUDE_CODE_PROVIDERS


def supports_sampling(provider: str) -> bool:
    """False when the provider ignores temperature/top_p."""
    return not is_claude_code(provider)


# Seconds without any output before a Claude Code call is killed.
CLAUDE_CODE_IDLE_TIMEOUT = 90
# Neutral cwd so the CLI never discovers a project CLAUDE.md or settings.
_CLAUDE_CODE_CWD = tempfile.gettempdir()


def claude_code_bare() -> bool:
    return os.environ.get("CLAUDE_CODE_BARE", "").strip().lower() in {"1", "true", "yes", "on"}


def claude_code_leak_terms() -> tuple[str, ...]:
    """Account identifiers Claude Code will attach to every call (empty in
    bare mode). The email's local part leaks on its own, without the domain,
    so it is matched separately from the generic email regex."""
    if claude_code_bare():
        return ()
    try:
        out = subprocess.run(
            [os.environ.get("CLAUDE_CODE_BIN", "claude"), "auth", "status", "--json"],
            capture_output=True, text=True, timeout=20, cwd=_CLAUDE_CODE_CWD,
        )
        email = (json.loads(out.stdout).get("email") or "").strip()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return ()
    if not email:
        return ()
    local = email.split("@", 1)[0]
    return (email, local) if len(local) >= 4 else (email,)


def claude_code_command(name: str, system_prompt: str) -> list[str]:
    cmd = [
        os.environ.get("CLAUDE_CODE_BIN", "claude"),
        "-p",
        "--bare" if claude_code_bare() else "--safe-mode",
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        # a non-default permission mode adds its own reminder to the context
        "--permission-mode", "default",
        "--effort", "low",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--model", name,
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    return cmd


# How far past the step budget the stream may run looking for a sentence end.
_SOFT_BUDGET_OVERRUN = 1.5
_SENTENCE_END_RE = re.compile(r"[.!?…](?:[\"'”’)*_]*)(?=\s|$)|\n")


def _clip_at_budget(
    text: str, emitted: int, budget: int, hard_cap: int
) -> tuple[str, bool]:
    """Trim a delta against the step budget. Returns (text_to_emit, stop).

    Below budget, text passes through. Once the budget is reached, emit up to
    and including the first sentence end (a newline counts) plus a separating
    space, then stop. At the
    hard cap, cut at the last whitespace so no word is ever split."""
    end = emitted + len(text)
    if end <= budget:
        return text, False
    search_from = max(0, budget - emitted)
    m = _SENTENCE_END_RE.search(text, search_from)
    if m and emitted + m.end() <= hard_cap:
        clipped = text[: m.end()]
        # The next step opens a fresh sentence with no leading space, so
        # leave the separator in place ("watched.Both" otherwise).
        return (clipped if clipped.endswith("\n") else clipped + " "), True
    if end <= hard_cap:
        return text, False
    room = text[: hard_cap - emitted]
    space = room.rfind(" ")
    if space > 0:
        return room[:space], True
    # No whitespace in this delta before the cap; the previous delta ended
    # on (or inside) a word already, so stop without adding a fragment.
    return "", True


def parse_claude_code_event(line: str) -> tuple[Optional[str], Optional[dict]]:
    """Parse one stream-json line into (text_delta, result_event).

    Raises RuntimeError on an error result so the caller's retry/backoff
    path sees it like any other provider failure."""
    line = line.strip()
    if not line:
        return None, None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None, None
    kind = event.get("type")
    if kind == "stream_event":
        inner = event.get("event") or {}
        delta = inner.get("delta") or {}
        if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
            return delta.get("text") or None, None
        return None, None
    if kind == "result":
        if event.get("is_error") or event.get("subtype") not in (None, "success"):
            detail = event.get("result") or event.get("errors") or event.get("subtype")
            raise RuntimeError(f"claude code: {detail}")
        return None, event
    return None, None


def _stream_claude_code(
    name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    tracker: Optional[UsageTracker],
) -> Iterator[str]:
    proc = subprocess.Popen(
        claude_code_command(name, system_prompt),
        env={**os.environ, "MAX_THINKING_TOKENS": "0"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=_CLAUDE_CODE_CWD,
    )
    # stdin is written from a thread so a large prompt can't deadlock
    # against a full stdout pipe.
    def _feed():
        try:
            proc.stdin.write(user_prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    threading.Thread(target=_feed, daemon=True).start()

    watchdog: list[threading.Timer] = []

    def _arm():
        if watchdog:
            watchdog[0].cancel()
            watchdog.clear()
        t = threading.Timer(CLAUDE_CODE_IDLE_TIMEOUT, proc.kill)
        t.daemon = True
        t.start()
        watchdog.append(t)

    # No max_tokens flag exists; approximate with the same 4-chars/token
    # heuristic used elsewhere. The budget is soft: an instruct model handed
    # a buffer that ends mid-sentence restarts that sentence instead of
    # continuing it ("keeps forgeThe hallway outside keeps forgetting"), so
    # once the budget is spent the stream runs on to the next sentence end.
    # Past the hard cap it stops at the last word boundary instead.
    char_budget = max(1, max_tokens) * 4
    hard_cap = int(char_budget * _SOFT_BUDGET_OVERRUN)
    emitted: list[str] = []
    emitted_chars = 0
    result: Optional[dict] = None
    cut = False
    try:
        _arm()
        for line in proc.stdout:
            _arm()
            text, res = parse_claude_code_event(line)
            if res is not None:
                result = res
            if not text:
                continue
            text, cut = _clip_at_budget(text, emitted_chars, char_budget, hard_cap)
            if text:
                emitted.append(text)
                emitted_chars += len(text)
                yield text
            if cut:
                break
    finally:
        if watchdog:
            watchdog[0].cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        for f in (proc.stdout, proc.stderr):
            if f:
                f.close()

    if not cut and result is None:
        raise RuntimeError(
            f"claude code exited {proc.returncode} without a result: {stderr.strip()[-300:]}"
        )

    if tracker is not None:
        usage = (result or {}).get("usage") or {}
        if usage:
            prompt_tokens = (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
            )
            tracker.add(
                prompt_tokens,
                int(usage.get("output_tokens") or 0),
                name,
                cost=result.get("total_cost_usd"),
            )
        else:
            # Cut early: the CLI never reached its result event.
            tracker.add_approx(system_prompt + user_prompt, "".join(emitted), name)


def stream_completion(
    provider: str,
    name: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    tracker: Optional[UsageTracker] = None,
) -> Iterator[str]:
    """Yields token strings as they stream in. If a UsageTracker is passed,
    accumulates usage from the stream's final chunk (preferred) or from a
    char/4 approximation (fallback when the provider omits usage)."""
    if is_claude_code(provider):
        if mode == "base":
            raise ValueError("claude_code provider has no base (raw completion) mode")
        yield from _stream_claude_code(name, system_prompt, user_prompt, max_tokens, tracker)
        return

    model = _model_string(provider, name, mode)
    sampling = {"temperature": temperature, "top_p": top_p}
    if provider == "anthropic" or model.startswith("claude-"):
        sampling = {"temperature": max(0.0, min(_ANTHROPIC_TEMP_MAX, temperature))}

    if mode == "base":
        # Base models have no chat template — concatenate any system framing
        # directly with the buffer. The seed/prompt design in prompts.py is
        # already written to bootstrap a base model from a non-instructional
        # prefix, so most callers will pass system_prompt='' here.
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = litellm.text_completion(
            model=model,
            prompt=prompt,
            **sampling,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        prompt_for_approx = prompt
    else:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **sampling,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        prompt_for_approx = f"{system_prompt}\n\n{user_prompt}"

    completion_buf: list[str] = []
    measured: Optional[tuple[int, int]] = None

    for chunk in response:
        delta = _extract_delta(chunk)
        if delta:
            completion_buf.append(delta)
            yield delta
        usage = _extract_usage(chunk)
        if usage is not None:
            measured = usage

    if tracker is not None:
        if measured is not None:
            tracker.add(measured[0], measured[1], model)
        else:
            tracker.add_approx(prompt_for_approx, "".join(completion_buf), model)


def complete_once(
    provider: str,
    name: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    tracker: Optional[UsageTracker] = None,
) -> str:
    """Non-streaming single-shot completion. Used by self_state for short
    auxiliary calls (e.g. summarization on phase transition)."""
    return "".join(
        stream_completion(
            provider=provider,
            name=name,
            mode=mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tracker=tracker,
        )
    )
