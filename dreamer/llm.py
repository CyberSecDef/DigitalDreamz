"""Multi-provider streaming via litellm.

Two prompting modes:
- 'instruct': chat-template path (system + user messages). For chat/instruct
  models. Uses litellm.completion.
- 'base': raw text-completion path (no chat template, buffer sent as prefix).
  For non-chat-tuned base models. Uses litellm.text_completion. For Ollama,
  this routes through /api/generate rather than /api/chat.
"""
import os
from typing import Iterator, Optional
import litellm

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

    def add(self, prompt_tokens: int, completion_tokens: int, model: str, approx: bool = False) -> None:
        self.prompt_total += prompt_tokens
        self.completion_total += completion_tokens
        self.prompt_delta += prompt_tokens
        self.completion_delta += completion_tokens
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
    model = _model_string(provider, name, mode)

    if mode == "base":
        # Base models have no chat template — concatenate any system framing
        # directly with the buffer. The seed/prompt design in prompts.py is
        # already written to bootstrap a base model from a non-instructional
        # prefix, so most callers will pass system_prompt='' here.
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = litellm.text_completion(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
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
            temperature=temperature,
            top_p=top_p,
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
