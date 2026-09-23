# dreamer

A small instrument for watching what an LLM does when nothing is asked of it.

Self-prompting loop with sleep-cycle phases (temperature oscillation where the provider supports it), fed by three input corpora (today's conversations, the day's news, and a curated dream-register substrate). Drift back into chat-assistant register triggers in-place "recovery surgery" — the buffer is wrapped as fading background and a fresh fragment is surfaced as the new continuation point. Sessions can self-extend the latent corpus across runs, so the dream remembers itself across nights. Streams tokens to the terminal in colored prose with a faint gutter for system events; logs everything to SQLite.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=...' >> .env.dev   # untracked overlay
.venv/bin/python -m dreamer.main           # uses .env and .env.dev (default)
```

CLI flags: `--env dev|prod` selects an `.env.<env>` overlay, `--perspective third|none` overrides the prompt mode, `--duration N` overrides session minutes. Precedence is shell environment > `.env.<env>` > `.env`, so `MODEL_NAME=x python -m dreamer.main` works for one-off runs.

Tests: `pip install -r requirements-dev.txt && python -m pytest`.

## Models

**Default: Claude via Claude Code** (`MODEL_PROVIDER=claude_code`, `MODEL_NAME=claude-sonnet-5`). Each step runs `claude -p` with no tools, no thinking, no session persistence, and the dream system prompt in place of Claude Code's. Auxiliary calls (self-state summaries, session distillation) go to `claude-haiku-4-5-20251001`; override with `AUX_MODEL_PROVIDER` / `AUX_MODEL_NAME`.

Trade-offs of this path:

- **No sampling controls.** The CLI exposes no temperature, top_p or max_tokens. The phase curve is carried by a one-line per-phase prose hint appended to the system prompt (`prompts.PHASE_HINTS`); `max_tokens_per_step` is enforced by cutting the stream. The `temperature` column in `tokens` records the nominal curve, not what was sampled.
- **`CLAUDE_CODE_BARE=true` (default) needs `ANTHROPIC_API_KEY`.** A logged-in Claude Code session attaches the account email, working directory, git status, model identity and date to every call, and no flag removes them. The dream prompt reads anything unexplained as residue, so the model writes that context straight into the dream (observed on every test call). `--bare` strips all of it except one line (`Today's date is …`, which `CLAUDE_CODE_OVERRIDE_DATE` does not change), so a date detector (`sampler.detect_date_leak`) cuts any full ISO date (`2026-09-22`) or "today's date is" phrasing with recovery surgery on every Claude Code run. Bare years like "1911" are left alone, so the current year can still surface on its own. It only authenticates with an API key.
- With `CLAUDE_CODE_BARE=false` the logged-in account is used and a harness-leak detector (`sampler.detect_harness_leak`) treats any email, system path, git/workspace or model-name mention as contamination and cuts it with recovery surgery. Expect frequent recoveries; the raw `tokens` table still records the leaked text before the cut.
- Roughly 2–3 s of CLI startup per step.

**Other providers** go through litellm: `MODEL_PROVIDER=ollama` / `openai` / `anthropic` with real temperature control (Anthropic clamps to 1.0 and drops `top_p`). For local base models set `MODEL_PROVIDER=ollama`, `MODEL_MODE=base`, and point `AUX_MODEL_*` at an instruct model, since a base model can't follow the summarize/distill instructions.

## The cycle

Every session is a series of phase arcs. One arc is `drift → light → deep → rem → surface`, configured via `SESSION_CYCLE_MINUTES` (default 15 min). Sampling temperature rises and falls with the phase; REM holds a high-temperature plateau, deep ramps up to peak weirdness, surface cools back down.

Phase boundaries (fractions of one cycle, in `sampler.py`):

| Phase   | Range       | Width | Notes                          |
|---------|-------------|-------|--------------------------------|
| drift   | 0.00 – 0.10 | 10%   | Settling — temp climbs to base |
| light   | 0.10 – 0.30 | 20%   | Base temp with small jitter    |
| deep    | 0.30 – 0.45 | 15%   | Temp ramps to max              |
| rem     | 0.45 – 0.85 | 40%   | High-temp plateau, vivid       |
| surface | 0.85 – 1.00 | 15%   | Cooling toward waking          |

`SESSION_DURATION_MINUTES` controls total runtime; the cycle repeats throughout. Slow models benefit from longer cycles (try 25+ min) so each phase has wall-clock room to develop.

## Corpus

Three input streams, each weighted per phase via `INJECTION_WEIGHTS_<PHASE>`:

- `corpus/today/` — drop today's conversation transcripts as `.txt`, `.md`, or `.log`. Day residue extractor pulls salient sentences (questions, named-entity-heavy lines).
- `corpus/latent/` — slow substrate. The repo ships a starter pool (~200 fragments across cities, rooms, fragments, transformations, waterworks, mirrors, time, light, objects, animals). Drop your own books, journals, and short prose alongside; per-file weights live in `corpus/latent/weights.txt`. Sampling picks a file by weight, then a whole paragraph (or line, for one-fragment-per-line files); paragraphs longer than `CORPUS_LATENT_CHUNK_CHARS` yield a run of whole sentences.
- World events — RSS feeds in `CORPUS_WORLD_FEEDS`, sanitized of URLs/datelines/wire-service residue before injection. Fetched with a 10 s per-feed timeout; refreshes after the first run in a background thread so the dream loop never blocks on the network.

Injections fire on stalling (n-gram overlap exceeds threshold) or on a timed interval with jitter. `INJECTION_MODE=visible` appends the fragment to the end of the buffer before the next generation; `deferred` holds it through that generation and then splices it in *behind* the text just written, so the model continues from its own words with the fragment as background.

## Accretion (latent corpus self-extension)

When `ACCRETION_ENABLED=true`, the latent corpus grows from each session in three auto-written tiers plus one manual:

- `corpus/latent/fixations/` (weight 0, kept on disk only) — captured online when stickiness recovery fires; raw "the model couldn't let this go" snippets.
- `corpus/latent/phase-summaries/` (weight 0, kept on disk only) — persists self-state summaries on every phase transition (requires `SELF_STATE_ENABLED=true`). Survives early termination — these checkpoints land before the post-session distillation step.
- `corpus/latent/sessions/` (weight 0.5) — post-session distillation via one extra LLM call over the session's surviving model text: injections and anything recovery surgery removed are excluded, so contaminated register never gets written back. Skipped on Ctrl-C, so this is the "best" version when the session completes normally.
- `corpus/latent/preserved/` (weight 1.0) — manual. Hand-promote anything you want to keep into the permanent substrate.

FIFO size caps (`ACCRETION_FIXATIONS_MAX`, `ACCRETION_PHASE_SUMMARIES_MAX`, `ACCRETION_DISTILLATIONS_MAX`) trim oldest auto-written files at session end so the substrate doesn't grow unbounded. All accreted text has `‹ ›` markers stripped before it is written, so a sampled file can't produce nested brackets.

## Recovery surgery

When the model drifts back into chat-assistant register — "Here's a list," "Let me explain," numbered lists, markdown headings, second-person `you'll`/`you've`/etc. — `sampler.detect_register_drift` fires. Phrases are word-boundary matched (so "there's a" and "violet meadow" don't trip "Here's a" / "Let me"), and curly apostrophes are normalized. The same machinery handles topical drift via a blocklist (`data/topical_blocklist.txt`) and, on logged-in Claude Code, harness leaks.

Detectors scan the model's own text only: `‹…›` fragments are scrubbed *before* the tail is sliced, including half-fragments cut by the window edge. The scan covers the whole of the latest step's output plus 100 chars of overlap (minimum 300), so drift at the start of a long step isn't missed.

On a hit, the buffer is truncated at the last clean sentence boundary (never inside an injected fragment), wrapped as `‹receding› ... ‹/receding›`, and a fresh latent fragment is appended as the new continuation point. The wrap tells the model that the prior text is fading background — material to drift onward from, not a tail to copy verbatim.

A separate softer signal, `register_stickiness`, fires when most of the model's recent content words are recycled from the preceding history for `STICKINESS_PATIENCE` consecutive steps. It triggers the same truncate-and-wrap surgery and writes a fixation file. Only text written since the last recovery is scored, so the history that recovery just pushed into the receding block can't re-trigger it straight away.

## Reading the terminal

Two redundant signals separate the dream from the system:

- **Gutter:** every system event prints with a `│ ` prefix in dim white. Model tokens flow flush-left in phase-colored prose.
- **Brackets:** `[square brackets]` in the token stream are the model's own voice (an established stylistic register from the system prompt). `‹angle brackets›` are anything the system injected — fragments, recovery markers, receding wraps, UI sigils.

Sigils in use:

| Sigil           | Event                                                        |
|-----------------|--------------------------------------------------------------|
| `── from → to ──` | Phase transition                                            |
| `‹src\|trg: …›` | Fragment injected from corpus / day residue / world events  |
| `⟂ drift`       | Register or topical drift logged                             |
| `⟂ recovered`   | Drift recovered: buffer truncated, recovery fragment injected|
| `↻ recovery`    | The recovery fragment surfacing                              |
| `‹receding ↓›` / `‹↑ surface›` | Buffer wrap markers framing a recovery        |
| `◇ self`        | Self-state summary refreshed (if `SELF_STATE_ENABLED`)       |
| `✎`             | Accretion event: fixation, phase-summary, distillation, prune|
| `⚠`             | Error                                                        |

A legend prints once at session start. Pipe a transcript through `grep -v '^│'` for the model output alone, or `grep '^│'` for system events alone.

## Logging

Every session writes to `dreams.db` (SQLite, WAL mode, one commit per step). Tables:

- `sessions` — one row per run, with config snapshot
- `tokens` — every token streamed, with timestamp, temperature, phase, step
- `injections` — every fragment injected, with source, trigger reason, content
- `phase_transitions` — phase changes with cycle position and window size
- `contamination_events` — register/topical/stickiness/harness/date hits with action taken
- `self_states` — phase-change summaries if `SELF_STATE_ENABLED=true`

Cross-session questions worth asking the data:

- What concepts recur across sessions independent of injections?
- What gets injected and never picked up vs. injected and metabolized?
- Does temperature correlate with concept fusion rate?
- What does the model never touch even when seeded?
- Which fixations/phase-summaries/distillations end up promoted to `preserved/`, and what do they share?

## Modes

`SESSION_PERSPECTIVE`:
- `third` — third person, no "I". More dreamlike narration with figures and rooms.
- `none` — no subject at all. Fragments, images, verbs without actors. More austere.

`MODEL_MODE`:
- `instruct` — chat-template path (system + user messages). For chat/instruct-tuned models.
- `base` — raw text-completion path. Buffer sent as prefix without chat scaffolding. For non-chat-tuned base models (litellm providers only). Self-state is skipped in base mode unless accretion is persisting it, since the summary never reaches a base model.

Run combinations as parallel conditions over time.
