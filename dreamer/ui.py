"""Terminal renderer. Tokens stream live, color-mapped to phase + temperature."""
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

    def render_injection(self, source: str, fragment: str, trigger: str):
        self.console.print(
            Text(f"\n[{source}|{trigger}: {fragment}]\n", style="dim cyan italic"),
            end="",
            soft_wrap=True,
        )

    def render_phase_change(self, from_phase: str, to_phase: str):
        self.console.print(
            Text(f"\n  ── {from_phase} → {to_phase} ──\n", style="dim white"),
            end="",
            soft_wrap=True,
        )

    def render_session_start(self, session_id: int, model: str, perspective: str):
        self.console.rule(f"[bold]dream session #{session_id}[/]  {model} · {perspective}")

    def render_session_end(self):
        self.console.rule("[bold]waking[/]")
