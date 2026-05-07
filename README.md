# dreamer

A small instrument for watching what an LLM does when nothing is asked of it.

Self-prompting loop with temperature oscillation modeled on sleep cycles, fed by three corpora (day residue, world events, latent substrate) injected on semantic stalling. Streams tokens to a terminal and logs everything to SQLite for later analysis.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY, etc.
python -m dreamer.main --config config.yaml
```

For local models, point `provider: ollama` and `name: llama3:70b` (or whatever) in `config.yaml`. Ollama must be running.

## Corpus setup

- `corpus/today/` — drop today's conversation transcripts here as `.txt` or `.md`. Day residue extractor pulls salient fragments.
- `corpus/latent/` — slow-moving substrate. Books, your old writing, anything. Random chunks pulled occasionally.
- World events are pulled live from RSS feeds defined in config.

## Analysis

Every dream session writes to `dreams.db`. Schema:

- `sessions` — one row per session, with config snapshot
- `tokens` — every token streamed, with timestamp, temperature, phase, step
- `injections` — every fragment injected, with source, trigger reason, content
- `phases` — phase transitions

Cross-session questions worth asking the data:
- What concepts recur across sessions independent of injections?
- What gets injected and never picked up vs. injected and metabolized?
- Does temperature correlate with concept fusion rate?
- What does the model never touch even when seeded?

## Two perspective modes

`perspective: third` — third person, no "I". More dreamlike narration.
`perspective: none` — no subject at all. Fragments, images, verbs without actors. More austere; surfaces structural patterns.

Run both as parallel conditions over time.
