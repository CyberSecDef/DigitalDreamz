"""Thin self-state thread — gives the dreamer a continuity hint across
generations by summarizing recent dream-text and prepending it to the system
prompt.

Triggered on phase transitions only (cheap-ish; small extra LLM call per
~3-5 minutes of dream-time). Disabled by default — toggle with
SELF_STATE_ENABLED=true.
"""
from . import llm


SUMMARY_PROMPT = (
    "Summarize the recurring imagery, mood, and themes in the following "
    "dream-text in 2-3 sentences, in the same register as the text itself. "
    "No preamble. No second person. No bullet points."
)

# Hard cap on summary length so the prepend doesn't dominate the prompt.
_MAX_SUMMARY_CHARS = 800


class SelfState:
    """Holds the current self-summary and knows how to refresh it."""

    def __init__(self, model_cfg: dict, sampling_cfg: dict, enabled: bool = False, tracker=None):
        self.model_cfg = model_cfg
        self.sampling_cfg = sampling_cfg
        self.enabled = enabled
        self.summary: str = ""
        self.tracker = tracker

    def ambient_prefix(self) -> str:
        """String to prepend to the system prompt for this generation."""
        if not self.enabled or not self.summary:
            return ""
        return f"Recent dream-state: {self.summary}\n\n"

    def refresh(self, recent_text: str) -> str:
        """Run a one-shot summary call against the tail of the buffer.

        Returns the new summary (also stored on self). Caller is responsible
        for persisting via DB.log_self_state.
        """
        if not self.enabled:
            return self.summary

        tail = recent_text[-2000:] if len(recent_text) > 2000 else recent_text
        if not tail.strip():
            return self.summary

        # Always run the summary call as instruct — even when the dreaming
        # model is in base mode, we need a chat-style instruction here. If
        # the configured model has no instruct path, this will fail loudly,
        # which is the right signal.
        try:
            text = llm.complete_once(
                provider=self.model_cfg["provider"],
                name=self.model_cfg["name"],
                mode="instruct",
                system_prompt=SUMMARY_PROMPT,
                user_prompt=tail,
                temperature=0.6,
                top_p=0.9,
                max_tokens=200,
                tracker=self.tracker,
            )
        except Exception:
            return self.summary

        text = text.strip()
        if len(text) > _MAX_SUMMARY_CHARS:
            text = text[:_MAX_SUMMARY_CHARS].rstrip() + "…"
        self.summary = text
        return self.summary
