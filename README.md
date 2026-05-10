# dreamer

A small instrument for watching what an LLM does when nothing is asked of it.

Self-prompting loop with temperature oscillation modeled on sleep cycles, fed by three input corpora (today's conversations, the day's news, and a curated dream-register substrate). Drift back into chat-assistant register triggers in-place "recovery surgery" — the buffer is wrapped as fading background and a fresh fragment is surfaced as the new continuation point. Sessions can self-extend the latent corpus across runs, so the dream remembers itself across nights. Streams tokens to the terminal in colored prose with a faint gutter for system events; logs everything to SQLite.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY, OLLAMA_API_BASE, etc.
python -m dreamer.main             # uses .env and .env.dev (default)
```

CLI flags: `--env dev|prod` selects an `.env.<env>` overlay, `--perspective third|none` overrides the prompt mode, `--duration N` overrides session minutes.

For local models: set `MODEL_PROVIDER=ollama` and `MODEL_NAME=llama3:70b` (or whatever) in `.env`. Ollama must be running. Set `MODEL_MODE=base` for non-chat-tuned base models.

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
- `corpus/latent/` — slow substrate. The repo ships a starter pool (~200 fragments across cities, rooms, fragments, transformations, waterworks, mirrors, time, light, objects, animals). Drop your own books, journals, and short prose alongside; per-file weights live in `corpus/latent/weights.txt`.
- World events — RSS feeds in `CORPUS_WORLD_FEEDS`, sanitized of URLs/datelines/wire-service residue before injection.

Injections fire on stalling (n-gram overlap exceeds threshold) or on a timed interval with jitter.

## Accretion (latent corpus self-extension)

When `ACCRETION_ENABLED=true`, the latent corpus grows from each session in three auto-written tiers plus one manual:

- `corpus/latent/fixations/` (weight 0.1) — captured online when stickiness recovery fires; raw "the model couldn't let this go" snippets.
- `corpus/latent/phase-summaries/` (weight 0.3) — persists self-state summaries on every phase transition (requires `SELF_STATE_ENABLED=true`). Survives early termination — these checkpoints land before the post-session distillation step.
- `corpus/latent/sessions/` (weight 0.5) — post-session distillation via one extra LLM call. Skipped on Ctrl-C, so this is the "best" version when the session completes normally.
- `corpus/latent/preserved/` (weight 1.0) — manual. Hand-promote anything you want to keep into the permanent substrate.

FIFO size caps (`ACCRETION_FIXATIONS_MAX`, `ACCRETION_PHASE_SUMMARIES_MAX`, `ACCRETION_DISTILLATIONS_MAX`) trim oldest auto-written files at session end so the substrate doesn't grow unbounded.

## Recovery surgery

When the model drifts back into chat-assistant register — "Here's a list," "Let me explain," numbered lists, markdown headings, second-person `you'll`/`you've`/etc. — `sampler.detect_register_drift` fires. The same machinery handles topical drift via a blocklist (`data/topical_blocklist.txt`).

On a hit, the buffer is truncated at the last clean sentence boundary, wrapped as `‹receding› ... ‹/receding›`, and a fresh latent fragment is appended as the new continuation point. The wrap tells the model that the prior text is fading background — material to drift onward from, not a tail to copy verbatim.

A separate softer signal, `register_stickiness`, fires when the model has been generating the same character-n-gram distribution for a long time even though surface tokens differ. That triggers a perturbation injection (no truncation) and writes a fixation file.

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

Every session writes to `dreams.db` (SQLite). Tables:

- `sessions` — one row per run, with config snapshot
- `tokens` — every token streamed, with timestamp, temperature, phase, step
- `injections` — every fragment injected, with source, trigger reason, content
- `phase_transitions` — phase changes with cycle position and window size
- `contamination_events` — drift/topical/stickiness hits with action taken
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
- `base` — raw text-completion path. Buffer sent as prefix without chat scaffolding. For non-chat-tuned base models.

Run combinations as parallel conditions over time.
