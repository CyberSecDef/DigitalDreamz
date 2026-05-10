"""Terminal renderer. Tokens stream live, color-mapped to phase + temperature.

Model tokens flow flush-left without markers — they are the dream itself.
Every non-token event (phase, injection, contamination, recovery, self-state,
errors) prints on its own line with a faint │ gutter, so system text is
visually filterable from the dream stream at a glance.
"""
from rich.console import Console
from rich.text import Text


PHASE_COLORS = {
    "drift":   "rgb(150,180,200)",   # pale blue
    "light":   "rgb(180,200,180)",   # pale green
    "deep":    "rgb(140,100,180)",   # violet (heavy)
    "rem":     "rgb(220,160,100)",   # warm amber (vivid)
    "surface": "rgb(200,200,160)",   # pale gold
}


class DreamRenderer:
    def __init__(self):
        self.console = Console()

    def render_token(self, token: str, phase: str, temperature: float):
        color = PHASE_COLORS.get(phase, "white")
        # higher temp → dimmer / italic feel
        style = color
        if temperature > 1.5:
            style = f"italic {color}"
        if temperature > 1.65:
            style = f"italic dim {color}"
        self.console.print(Text(token, style=style), end="", soft_wrap=True)

    def _print_system(self, content: str, style: str):
        """Render a non-token event with a faint gutter prefix on every line,
        bracketed by a leading newline so it breaks cleanly from the token
        stream and leaves the cursor on a fresh line for the next token."""
        self.console.print()
        for line in content.split("\n"):
            self.console.print(
                Text.assemble(("│ ", "dim white"), (line, style)),
                soft_wrap=True,
            )

    def render_injection(self, source: str, fragment: str, trigger: str):
        self._print_system(f"‹{source}|{trigger}: {fragment}›", "dim cyan italic")

    def render_contamination(self, pattern: str, action: str):
        style = "bold red" if action == "recovered" else "yellow"
        label = "recovered" if action == "recovered" else "drift"
        self._print_system(f"⟂ {label}: {pattern} ⟂", style)

    def render_recovery(self, fragment: str):
        """Visible signal for a recovery fragment that was appended to the
        buffer after a contamination — previously silent."""
        self._print_system(f"↻ recovery: {fragment}", "dim cyan italic")

    def render_receding_open(self):
        """Marks the moment the prior buffer prefix was wrapped as receding
        background. Pairs with render_receding_close around the recovery."""
        self._print_system("‹receding ↓›", "dim white")

    def render_receding_close(self):
        self._print_system("‹↑ surface›", "dim white")

    def render_self_state(self, summary: str):
        self._print_system(f"◇ self ◇  {summary}", "dim magenta italic")

    def render_accretion(self, message: str):
        """Latent-corpus self-extension events (fixation captured, session
        distilled, prune cleanup)."""
        self._print_system(f"✎ {message}", "dim green")

    def render_phase_change(self, from_phase: str, to_phase: str):
        self._print_system(f"── {from_phase} → {to_phase} ──", "dim white")

    def render_error(self, message: str):
        self._print_system(f"⚠ {message}", "red")

    def render_interrupt(self):
        self._print_system("interrupted", "dim")

    def render_session_start(self, session_id: int, model: str, perspective: str):
        self.console.rule(f"[bold]dream session #{session_id}[/]  {model} · {perspective}")

    def render_legend(self, self_state_enabled: bool = False, accretion_enabled: bool = False):
        """Print a one-time key explaining the system markers used during
        the session. Called once after session start so the user can read
        the rendered stream without having to remember what each sigil means."""
        entries: list[tuple[str, str, str]] = [
            ("── phase change",         "phase transition: drift · light · deep · rem · surface",       "dim white"),
            ("‹src|trg: fragment›",     "fragment injected from corpus / day residue / world events",   "dim cyan italic"),
            ("⟂ drift",                 "model went off register or onto a blocked topic — logged",     "yellow"),
            ("⟂ recovered",             "drift recovered: buffer truncated, recovery fragment injected", "bold red"),
            ("↻ recovery",              "fragment surfacing as the new continuation point",             "dim cyan italic"),
            ("‹receding ↓› ‹↑ surface›", "prior buffer wrapped as fading background, fresh text rising", "dim white"),
        ]
        if self_state_enabled:
            entries.append(("◇ self",   "self-state summary refreshed (on phase change)", "dim magenta italic"))
        if accretion_enabled:
            entries.append(("✎",        "latent corpus self-extension: fixation, distillation, prune", "dim green"))
        entries.append(("⚠",            "error (API failure or system-level)",                          "red"))

        width = max(len(marker) for marker, _, _ in entries) + 2

        self.console.print()
        self.console.print(
            Text.assemble(("│ ", "dim white"), ("legend", "dim white bold")),
            soft_wrap=True,
        )
        for marker, desc, style in entries:
            self.console.print(
                Text.assemble(
                    ("│   ", "dim white"),
                    (f"{marker:<{width}}", style),
                    (desc, "dim white"),
                ),
                soft_wrap=True,
            )
        self.console.print(
            Text.assemble(
                ("│ ", "dim white"),
                ("‹…›", "dim cyan"),
                (" = system / injected text     ", "dim white"),
                ("[…]", "white"),
                (" = model's own voice", "dim white"),
            ),
            soft_wrap=True,
        )

    def render_session_end(self):
        self.console.rule("[bold]waking[/]")
