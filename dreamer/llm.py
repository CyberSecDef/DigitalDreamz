"""Multi-provider streaming via litellm.

litellm normalizes Anthropic / OpenAI / Ollama / Groq / etc. behind one call.
For Ollama, model string is `ollama/llama3:70b`. For Anthropic native,
just the model name (e.g. `claude-sonnet-4-5`) works if ANTHROPIC_API_KEY is set.
"""
from typing import Iterator
import litellm


def _model_string(provider: str, name: str) -> str:
    # litellm convention: provider/model unless it's the default for that provider
    if provider in {"anthropic", "openai"}:
        # litellm auto-routes if API key for that provider is present
        return name
    return f"{provider}/{name}"


def stream_completion(
    provider: str,
    name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Iterator[str]:
    """Yields token strings as they stream in."""
    model = _model_string(provider, name)
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
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            continue
        if delta:
            yield delta
