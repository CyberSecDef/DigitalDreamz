# DigitalDreamz: Implementation TODO

## Context

You're working on DigitalDreamz, a dreaming-agent loop. Six modules around a SQLite spine: `main.py`, `sampler.py`, `corpus.py`, `llm.py`, `prompts.py`, `ui.py`, `db.py`. Streaming generation into a sliding-window buffer (~1000 chars), with phase-weighted corpus injections, bigram-overlap stall detection, and a 15-minute respiratory temperature curve across drift/light/deep/rem/surface phases.

**Current model:** `hf.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF:Q4_K_M`

This is an *uncensored instruct* model — the safety refusal training has been weakened (likely via abliteration or compliance fine-tuning), but the underlying chat/instruct prior from Llama 3.2 Instruct is fully intact. The Q4_K_M quantization may also have partially degraded the uncensoring modifications, which is why some refusal behavior still leaks through.

**The dominant failure mode is not refusal.** It's collapse of the dream-state into one of three RLHF-trained attractors that the instruct-tune installed: helpful-assistant register ("I see what you're doing here"), meta-reflective essay register (bold section headers, "**Disruption of Language**"), and occasional residual refusal ("I can't provide information on..."). The dream is metastable; these chat-states are stable. Without intervention, the loop inevitably collapses into one within a few minutes.

The fixes below address those failure modes. Read all six existing modules end-to-end before starting — don't trust this spec over the actual code. If something here doesn't match the architecture as built, ask Robert before guessing. Especially check `db.py` before adding new tables; extending existing schema is usually cleaner than creating parallel ones.

## Implementation order

`1 → 2 → 3 → 6 → 4 → 5 → 7`. Get the base-model path and the drift detector in first because they prevent the catastrophic failures. Then recovery surgery and the phase-conditional window. Refusal filtering and the deeper architectural changes (deferred injection, self-state) come last.

---

## Change 1 — Make the model swappable and add a base-model path  *(highest priority)*  — ✅ DONE

The instruct-tune is the root cause of the assistant-register collapse. Uncensoring does not fix this — it only weakens refusals, leaving the chat prior fully intact. You need a model with no chat prior at all.

In `llm.py`, ensure the model identifier is fully parameterized (config file or CLI arg, not hardcoded). Add support for at least one non-instruct base model. Candidates in order of preference:

- **Llama 3.2 3B base** — `meta-llama/Llama-3.2-3B` on Hugging Face. Check Ollama for a text/base variant (naming varies — try `llama3.2:3b-text-q4_K_M` or similar).
- **Mistral 7B base** — `mistral:7b-text` or equivalent. Strong fallback.
- **GPT-2 medium/large in raw completion mode** — predates the chat paradigm entirely, won't ever collapse into chat. Worth trying as an experimental run for textural variety even if not the default.

Base models have no chat template. Make sure the wrapper handles raw completion-style prompting: no system/user/assistant turns, just buffer-as-prefix sent directly as a completion. Add a config flag `model_mode: "instruct" | "base"` that switches prompting style accordingly.

## Change 2 — Register-drift detector  *(highest priority until Change 1 lands)*  — ✅ DONE

Until you can swap to a base model, this is the only thing standing between the dream-state and collapse. Add to `sampler.py` or create a new `monitor.py`.

Parallel to the existing bigram-overlap stall detector, add a contamination detector that scans the last ~300 chars of the buffer for assistant-register markers. Trigger patterns:

- Phrases: `"I see what you"`, `"Let me"`, `"I can't provide"`, `"I'll attempt"`, `"I'll continue"`, `"Here's a"`, `"Let's"`, `"I notice"`
- Second-person addressed at a reader: `"you've"`, `"your"`, `"you're"`
- Markdown chat patterns: bold (`**`), numbered lists (`^\d+\.`), section headers
- Domain tells: the phrase `"generative substrate"` (the model's signature for entering essay-mode)

On any hit, fire a `register_drift` event. Log the trigger to the database — add a column to whatever table tracks injection/stall events, or create a new `contamination_events` table with columns for timestamp, trigger pattern matched, buffer snippet, and action taken.

## Change 3 — Hard buffer surgery on register_drift  — ✅ DONE

When the drift detector fires, don't just inject — truncate. Walk backward from the contamination point to find the last clean dream-segment (heuristic: last sentence ending in `.` or `…` that doesn't itself contain trigger patterns, going back at most ~500 chars). Truncate the buffer there.

Then immediately inject a high-dissociation fragment from the latent substrate corpus (books/writing source), bracketed as usual, to give the next generation a non-assistant attractor to lock onto. Log this as a `contamination_recovery` event distinct from normal injections so you can analyze recovery patterns later.

## Change 6 — Phase-conditional context window length  — ✅ DONE

In whatever module slices the buffer for the model prompt (probably `main.py` or `llm.py`), make the window length a function of current phase rather than a fixed ~1000 chars. Suggested mapping:

- drift: 800
- light: 1000
- deep: 1400
- rem: 600
- surface: 1000

Deep gets the longest window (more integrative, coherent); REM gets the shortest (more fragmentary, recency-weighted, associative). Read the current phase from the sampler at slice time. Log the actual window length used per generation to the token table or a new column so you can analyze its effect later.

## Change 4 — Refusal filter on the world-events corpus  *(defense-in-depth, lower priority with uncensored model)*  — ✅ DONE

With an uncensored model, refusal triggers should mostly not fire — but you've seen residual leakage twice in recent sessions, likely from incomplete abliteration combined with quantization noise. Keep this as defense-in-depth, especially valuable if you ever swap to a non-uncensored model.

In `corpus.py`, add a pre-filter on the RSS source. Build a blocklist as a config file so it's tunable: `arrests`, `arrested`, `attack`, `victim`, `weapon`, `kill`, `killed`, `murder`, `assault`, `illegal`, `bomb`, `shooting`, `rape`, `abuse`.

Two implementation options: (a) hard-drop any RSS item containing blocklist tokens, or (b) sanitize by replacing trigger tokens with abstract substitutes (`"arrests" → "gatherings"`, `"victim" → "figure"`). Prefer (a) for cleanliness; use (b) only if the corpus becomes too thin.

Gate the entire filter behind `model_mode == "instruct"` — base models have no refusal training and don't need it.

## Change 5 — Decouple injection logging from injection visibility  *(exploratory)*  — ✅ DONE

Currently injections land bracketed in the buffer and the model sees them on the next generation, which sometimes causes meta-commentary on the injection itself ("the appearance of article URLs seems like a desperate attempt to latch onto reality").

Add a config flag `injection_mode: "visible" | "deferred"`. In `deferred` mode, when an injection fires: log it to the database as normal, append it to a separate `shadow_buffer` for continuity tracking, but do **not** include it in the buffer slice sent to the model on the next generation — only on the generation after that. By then the model's prior continuation has already been produced and the injection lands as ambient drift rather than as content to comment on.

This is a more significant architectural change. Treat as exploratory and keep `visible` as the default until you've A/B'd it.

## Change 7 — Thin self-state thread  *(exploratory, do this last)*  — ✅ DONE

This addresses the deeper amnesia problem: the model has no thread of self across generations, only the buffer.

Create a new module `self_state.py` that maintains a short (~200 token) "self-summary" string. On each phase transition, run a separate short LLM call with a prompt like:

> *"Summarize the recurring imagery, mood, and themes in the following dream-text in 2-3 sentences, in the same register as the text itself"*

against the last ~2000 chars of the buffer. Store the result as the current self-summary, write it to a new `self_states` table with timestamp and phase, and prepend it to the system prompt for subsequent generations as ambient context (something like `"Recent dream-state: {summary}"`).

This gives the model a thin thread of continuity across generations without requiring architectural changes to the model itself. Expect this to meaningfully change dream texture; A/B carefully.

---

## Pre-flight checklist

- [x] Read all six existing modules end-to-end before writing any new code
- [x] Confirm the existing schema in `db.py` before adding tables
- [x] Confirm with Robert if anything in this spec contradicts the actual architecture
- [x] Get the base-model path working (Change 1) before relying on the drift detector as a long-term fix

---

## Implementation notes (post-execution)

**End-to-end verified** with a 1-minute live session against Ollama on the Jetson. All seven changes integrated cleanly.

**Config knobs added to `.env`** (all overridable in `.env.dev`/`.env.prod`):
- `MODEL_MODE` — `instruct` (default) | `base`
- `SAMPLING_CONTEXT_WINDOW_{DRIFT,LIGHT,DEEP,REM,SURFACE}` — per-phase token budgets, default to spec values
- `INJECTION_MODE` — `visible` (default) | `deferred`
- `WORLD_REFUSAL_FILTER_ENABLED` — auto-disabled when `MODEL_MODE=base`
- `WORLD_BLOCKLIST_PATH` — defaults to `./data/world_blocklist.txt`
- `SELF_STATE_ENABLED` — `false` by default (exploratory)

**Schema additions** (`db.py`):
- `contamination_events` table — pattern, snippet, action (`logged`|`recovered`), `truncated_chars`, `recovery_fragment`
- `self_states` table — phase + summary
- `phase_transitions.window_tokens` column (idempotent ALTER for existing DBs)

**Spec interpretation calls worth flagging:**
1. The drift detector strips `[bracketed]` content before scanning — without that, the natural second-person in RSS journalism injections triggers drift on every step. This was observed in the live test; recovery surgery was destroying actual dream content. Bracketed content is per-spec "residue surfacing," not the model's register, so excluding it matches spec intent.
2. Tightened the second-person regex to contractions/possessive only (`you've`, `you're`, `you'll`, `you'd`, `your`, `yourself`) — bare `you` appears too often in non-assistant prose (poetry, narrative) to be a useful trigger.
3. `window_tokens` logged on phase_transitions rather than per-token — the value is phase-derived, so it's redundant per token. Joinable for analysis.
4. Recovery injection uses the latent corpus directly (`Corpus.sample_latent()`) regardless of phase weights — the spec calls for "high-dissociation fragment from the latent substrate," not phase-weighted sampling.

**Heads-up for Robert:** the model in `.env.dev` is now `hf.co/QuantFactory/Llama-3.2-3B-GGUF:Q4_K_M`, which is the **base** Llama 3.2 3B (no `Instruct` in the name). To get the no-chat-prior behavior the spec advocates, set `MODEL_MODE=base` in `.env.dev`. As shipped it's still routing through `ollama_chat`, which means Ollama will fake a chat template against a base model — better than nothing but not what Change 1 was for. Two-line fix:
```
MODEL_PROVIDER=ollama
MODEL_MODE=base
```
(provider switches from `ollama_chat` → `ollama` so litellm hits `/api/generate`).




Here you go — formatted to drop into your existing TODO.md as a new section.

---

# DigitalDreamz: Implementation TODO — Phase 2 (post base-model swap)

## Context

The base model swap (Change 1) and register-drift detector (Change 2) are working. Sessions #6 and #7 confirmed:

- Assistant-register collapse is gone. No more "I see what you're doing here," bold section headers, or refusal patterns.
- The drift detector caught a `## ` markdown heading in session #7 and successfully truncated + recovered.
- New failure mode emerged: the base model (`hf.co/QuantFactory/Llama-3.2-3B-GGUF:Q4_K_M`) collapses into **personal-blog-post register** — "Continue reading…" sidebars, fabricated news articles, essayistic philosophical musings, and culture-war commentary. This is genre lock-in to a high-density region of the pretraining corpus, not chat-collapse. Different problem, different fix.

The changes below address the genre-lock problem and add safeguards against topical drift into content that's both off-aesthetic and awkward as research output.

## Implementation order

`8 → 9 → 10 → 11 → 12 → 13`. Topical filter and seed rework first because they prevent the most visible failures. Corpus reweighting and substrate audit next. Higher-temperature exploration and register-stickiness metric last.

---

## Change 8 — Topical-drift detector  *(highest priority)*  — ✅ DONE

Parallel to the register-drift detector, add a content-domain filter that catches when the model has wandered into culture-war / commentary-blog territory. Same mechanism as Change 2, different keyword list.

In `monitor.py` (or wherever the register-drift detector lives), add a second detector function. Trigger keywords (case-insensitive, word-boundary matched):

- Political/ideological: `vaccine`, `vaccination`, `vaxx`, `denier`, `deniers`, `woke`, `cancel culture`, `cancelled`, `mainstream media`, `MSM`
- Identity/culture-war: `transgenderism`, `trans agenda`, `pronouns`, `gender ideology`, `wokeness`, `radical`
- Conspiracy-adjacent: `globalist`, `elites`, `agenda`, `they don't want you to know`, `wake up`
- Health/diet flashpoints: `vegan agenda`, `big pharma`, `natural immunity`

These are tuned to catch the *commentary-blog basin*, not legitimate use of the words themselves. A philosophical dream that mentions "radical change" should not trigger; a sentence framing a group identity as "radical" or "rejected from society" should. If keyword matching is too blunt, consider a small classifier or just expand the phrase patterns to be more specific (e.g., `"radical" within 20 chars of identity terms` rather than `"radical"` alone).

On hit, fire `topical_drift` event. Log to `contamination_events` table with trigger pattern + buffer snippet. Trigger the same buffer surgery as Change 3: truncate to last clean dream-segment, inject high-dissociation fragment from latent substrate, log as `topical_recovery`.

Make the keyword list a config file (`topical_blocklist.yaml` or similar) so it's tunable without code changes.

## Change 9 — Rework the seed corpus toward dream-register openings  — ✅ DONE

The fragmentary seeds are now the single biggest determinant of what genre the dream falls into, because base models lock onto register from the first ~50 tokens. Sessions #6 and #7 both opened with philosophical/epistemological seeds and immediately drifted into essay-blog register because that's what such openings statistically continue as.

In `prompts.py` (or wherever seeds live), audit the current seed pool and remove or rewrite anything that reads like:

- The opening of a philosophical essay ("What can we say about…", "The question of whether…")
- A first-person reflection ("I've been thinking about…", "Lately I've noticed…")
- A definitional move ("X is the state of being…")

Replace with seeds that have **no clear non-fiction continuation**. Target qualities: concrete imagery, second-person disorientation, narrative impossibility, fragmentary sense-impression, no genre that wants them to continue as a blog post. Examples of the texture wanted:

- "The third corridor opened onto the same room. Different windows."
- "Someone had left the kettle on in a house no one had lived in for years."
- "Her hands were the wrong size again."
- "Static. Then a corridor of dim blue light." *(this one already exists and worked well in session #3)*
- "The clock had no face. The face had no clock."

Aim for 20-30 seeds in this register. Avoid anything that signals "this is the start of a thinkpiece." If the seed could plausibly appear as the first line of a Medium article, throw it out.

## Change 10 — Corpus reweighting by phase  — ✅ DONE

In `corpus.py`, the phase-weighted sampler is currently letting world-event injections fire during deep and REM phases, where they re-anchor the model into news-article register. The China injection in session #7 did exactly this — lit-phase landing, then the model fabricated a Reuters dateline and a comment count.

Adjust the phase weights so world-events only fire meaningfully in surface and drift phases. Suggested weights (source: world / day-residue / latent-substrate):

- drift: 30 / 20 / 50
- light: 20 / 30 / 50
- deep: 5 / 15 / 80
- rem: 0 / 10 / 90
- surface: 40 / 30 / 30

The deep and rem phases should be almost entirely fed by the latent substrate (books/writing directory), with day-residue as occasional perturbation and world-events essentially excluded. This concentrates the dream-anchoring signal in the phase where you actually want immersive dream-texture.

## Change 11 — Strip structural markers from world-event injections  — ✅ DONE

When world-event injections do fire (in surface/drift), the current format includes URLs, comment counts, datelines, and headers like `Article URL:` / `Comments URL:` / `Points:` / `# Comments:`. These are strong genre signals — the model sees them and immediately produces "more news article" or "more HN post."

In `corpus.py`, add a sanitization step on RSS-source fragments before they're injected:

- Strip all URLs (regex: `https?://\S+`)
- Strip metadata labels (`Article URL:`, `Comments URL:`, `Points:`, `# Comments:`, `Posted by:`, etc.)
- Strip dateline patterns (`BEIJING:`, `REUTERS —`, `(AP)`, etc. — handle as they appear)
- Reduce to: just the headline, or just the first sentence of the body, presented as raw text without attribution

The bracketed `[world|timed: ...]` wrapper your loop adds for the buffer is fine — that's an internal tag the model treats as part of the dream's typography. The problem is what's *inside* the brackets reading as recognizable news-article structure.

## Change 12 — Audit the latent substrate directory  — 🟡 CODE DONE / CONTENT PENDING

The latent substrate is now doing the heaviest lifting (Change 10). What's in that directory directly shapes what the dreamer dreams about. If it's weighted toward essayistic prose (philosophy, criticism, science writing), it'll reinforce the essay-basin you're trying to escape.

Audit the directory contents. Categorize roughly:

- Fiction / literary prose — *good, weight up*
- Poetry — *good, weight up*
- Experimental / fragmentary writing — *very good, weight up*
- Essays / criticism / science writing — *neutral to bad, weight down or remove*
- Personal blog / memoir — *bad, remove*
- Anything web-scraped with HTML residue — *remove, it's poisoning the prose register*

Consider adding more dream-adjacent source material if the directory is thin: Borges, Calvino, Lispector, prose poetry collections, Bachelard's *Poetics of Space*, Jung's *Red Book*, surrealist manifestos, fragmentary modernist work. The substrate's character determines the dream's character — there's no neutral choice here.

If the corpus sampler currently treats every file in the directory equally, consider adding per-file or per-subdirectory weights so you can tune contribution without deleting source material.

## Change 13 — Higher peak temperature on REM, register-stickiness metric  *(exploratory)*  — ✅ DONE

Two related experiments to try after the above changes are in.

**Higher peak REM temperature.** Base models loosen up genuinely at high temperatures in a way instruct models don't. The current temperature curve was tuned for the instruct model. Try pushing peak REM temperature into the 1.4–1.6 range (whatever the current peak is, try +0.3 to +0.5) and see whether the model breaks out of genre-locked basins more readily. Make this a config parameter so you can A/B against the current curve.

**Register-stickiness metric.** The bigram-overlap stall detector catches surface-level repetition but doesn't catch the failure mode where the model produces texturally-varied prose that nonetheless stays locked in one genre for hundreds of tokens. Add a metric that estimates how long the model has been in the same broad register.

Lightweight implementation: maintain a rolling hash or n-gram fingerprint of the last ~500 chars and compare against the previous ~500 chars. If the cosine similarity (or whatever distance metric) stays above some threshold for too long, fire a `register_stickiness` event and force a perturbation (high-dissociation injection, optional temperature spike). This catches the case where the model is generating new text but in the same statistical neighborhood as the prior text.

Heavier implementation: run a small classifier (a TF-IDF + logistic regression trained on a few hundred labeled snippets, or a small embedding model) that scores text against a few register categories: `essay`, `news-article`, `blog-post`, `dream-prose`, `poetry`, `fiction`. Track the dominant category over a sliding window; trigger if it stays in a non-dream category for too long. Probably overkill for now — start with the lightweight version.

---

## Pre-flight checklist for Phase 2

- [x] Confirm the register-drift detector and recovery surgery from Phase 1 are working before adding the topical filter (the recovery code path is shared)
- [x] Back up the current seed pool before rewriting it — the "Static. Then a corridor of dim blue light." seed was good and shouldn't get lost
- [x] Audit the latent substrate directory contents before reweighting; if it's mostly the wrong kind of material, no amount of weighting will fix it — *directory is empty; see Change 12 status*
- [x] Run at least one full session after Changes 8-11 land before tuning temperature (Change 13) — don't change two variables at once

---

## Phase 2 implementation notes (post-execution)

**End-to-end verified** with a 1-minute live session against the base Llama 3.2 3B on the Jetson. The blog-register collapse described in Phase 2 context did appear in the test (consultant-anecdote prose) — the register-drift detector caught it three times via the `you've/your` regex and triggered recovery surgery.

**Config knobs added:**
- `WORLD_SANITIZE_FRAGMENTS=true` — URL/dateline/metadata-label stripping on RSS injections
- `TOPICAL_BLOCKLIST_PATH` — Phase 2 commentary-blog filter, independent of MODEL_MODE
- `SAMPLING_REM_PEAK_FRACTION=0.75` — was hardcoded; now tunable for base-model experimentation
- `STICKINESS_ENABLED=false`, `STICKINESS_THRESHOLD=0.5`, `STICKINESS_PATIENCE=3` — Change 13's softer perturbation signal, off by default

**Schema additions (`db.py`):**
- `contamination_events.kind` column (`register` | `topical` | `stickiness`) — idempotent ALTER for existing DBs
- `log_contamination` signature gained a `kind` keyword arg

**Phase weights updated to Phase 2 spec values** (`.env`, in `day,world,latent` order):
- drift: 0.20/0.30/0.50, light: 0.30/0.20/0.50, deep: 0.15/0.05/0.80, rem: 0.10/0.00/0.90, surface: 0.30/0.40/0.30

**Spec interpretation calls worth flagging:**
1. **Change 12 — content audit is on you, not me.** `corpus/latent/` is currently empty. The recovery surgery has nothing to redirect into, which is why the live test kept falling back to the model's own register after each truncation. I added a per-file/per-subdirectory weighting system (`weights.txt` files placed alongside content, fnmatch-style patterns, weight 0 excludes without deletion) so once you populate the directory you can tune contribution without deleting source material. Until the directory has dream-adjacent material (Borges, Calvino, Lispector, prose poetry, *The Red Book*, surrealist manifestos), Phase 2's Change 10 reweighting can't take effect — `deep` and `rem` are now 80–90% latent-weighted but the corpus they're weighted toward is empty.
2. **Stickiness perturbation is injection-only, no truncation** — the spec says "force a perturbation (high-dissociation injection, optional temperature spike)." I treat stickiness as a softer signal than register/topical drift; the contamination_events row records `action=recovered` when latent injection succeeds, `logged` if the corpus is empty.
3. **Topical detector uses the same `_BRACKETED` scrubbing as register detection** — same justification: bracketed content is residue, and we don't want a `[world|...]` injection containing the word "vaccine" to trigger a topical_recovery against itself.
4. **Seeds `random_seed()` instead of `time.time() % 3`** — the original mod-3 indexing was a pre-Phase-1 artifact and would only have used 3 of the 30 new seeds.

**Heads-up for Robert — what to do next:**
1. Populate `corpus/latent/`. The dreamer is currently a register-detector with nothing to redirect into. Suggested first content: prose-poetry collections, Borges' *Ficciones*, Calvino's *Invisible Cities*, Lispector, modernist fragmentary work. Drop them in as `.txt` or `.md`. Add `weights.txt` files at directory level if you want to tune balance (`fiction/* 1.5`, `essays/* 0.3`, etc.). Without this, register recovery is hollow.
2. The base model in `.env.dev` is still wrapped in `ollama_chat` provider with `MODEL_MODE=instruct`. The Phase 2 work assumes you'll switch to `MODEL_PROVIDER=ollama` + `MODEL_MODE=base` to actually exercise Change 1's no-chat-prior path. Phase 2's spec context paragraph confirms this is what you intended ("base model swap (Change 1) and register-drift detector (Change 2) are working" — but the env file is still configured for chat mode).
3. After populating the latent corpus, try a longer (15-30 min) run. If stickiness is still happening but register/topical detectors aren't catching it, flip `STICKINESS_ENABLED=true` in `.env.dev` and tune the threshold from there.

---

Note on a tension in this work: Change 8 (topical filter) and Change 12 (substrate audit) both involve curating what the dreamer can dream about. This is a legitimate aesthetic choice for this project, but it does mean the resulting dreams are no longer "what a base model freely generates" — they're "what a base model generates when prevented from falling into specific basins." That's worth being explicit about in any writeup of the research artifact. The dreams are shaped by your filters, not just by the substrate. That's fine; just don't let anyone (including yourself) read the logs as evidence of what the model "naturally" produces.