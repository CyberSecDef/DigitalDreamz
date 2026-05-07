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


def stream_completion(
    provider: str,
    name: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Iterator[str]:
    """Yields token strings as they stream in."""
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
        )
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
        )

    for chunk in response:
        delta = _extract_delta(chunk)
        if delta:
            yield delta


def complete_once(
    provider: str,
    name: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
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
        )
    )
