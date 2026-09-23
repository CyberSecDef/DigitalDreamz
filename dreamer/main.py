"""Main dream loop: self-prompting, oscillating, injecting, logging."""
import os
import time
import random
import click

from . import prompts, llm, sampler, ui, db as db_mod, accretion
from .config import load_config
from .corpus import Corpus
from .self_state import SelfState


# The buffer only ever needs to feed the prompt window (≤ ~6k chars) and the
# stickiness metric (3.5k chars of model text). Collapse it past the high
# water mark so per-step joins and scans stay cheap in long sessions.
_BUFFER_KEEP_CHARS = 16000
_BUFFER_HIGH_WATER = 24000
# Drift scan covers at least this much model text, and always the whole of
# the latest step's output plus this much overlap.
_MIN_SCAN_CHARS = 300
_SCAN_OVERLAP_CHARS = 100
# Distillation reads at most 12k chars; keep a little more than that.
_TRANSCRIPT_KEEP_CHARS = 20000
_MAX_BACKOFF_SECONDS = 60


class Transcript:
    """The model's own surviving text: every streamed token, minus whatever
    recovery surgery removed. Injections never enter it. This is what the
    session distillation reads, so contaminated text doesn't get written back
    into the latent corpus."""

    def __init__(self):
        self._parts: list[str] = []
        self._chars = 0

    def append(self, token: str):
        self._parts.append(token)
        self._chars += len(token)
        if self._chars > _TRANSCRIPT_KEEP_CHARS * 2:
            self._parts = [self.text()[-_TRANSCRIPT_KEEP_CHARS:]]
            self._chars = len(self._parts[0])

    def drop(self, n: int):
        if n <= 0:
            return
        text = self.text()
        self._parts = [text[:-n] if n < len(text) else ""]
        self._chars = len(self._parts[0])

    def text(self) -> str:
        return "".join(self._parts)


def run_session(config: dict):
    sess_cfg = config["session"]
    model_cfg = config["model"]
    aux_cfg = config.get("aux_model", model_cfg)
    samp_cfg = config["sampling"]
    inj_cfg = config["injection"]
    log_cfg = config["logging"]
    mon_cfg = config.get("monitor", {})

    perspective = sess_cfg["perspective"]
    duration = sess_cfg["duration_minutes"] * 60
    cycle = sess_cfg["cycle_minutes"] * 60
    mode = model_cfg.get("mode", "instruct")
    injection_mode = inj_cfg.get("mode", "visible")
    window_by_phase = samp_cfg.get("context_window_by_phase", {})
    default_window = samp_cfg["context_window_tokens"]
    rem_peak = samp_cfg.get("rem_peak_fraction", 0.75)
    # Providers without sampling controls get the phase as prose instead.
    use_phase_hint = not llm.supports_sampling(model_cfg["provider"])
    # Logged-in Claude Code attaches account/workspace context to each call;
    # watch for it surfacing in the dream (see llm module docstring).
    watch_harness = llm.is_claude_code(model_cfg["provider"]) and not llm.claude_code_bare()
    leak_terms = llm.claude_code_leak_terms() if watch_harness else ()
    # The date line survives even --bare, so watch for it on every Claude Code run.
    watch_date = llm.is_claude_code(model_cfg["provider"])

    topical_patterns = sampler.load_topical_patterns(
        mon_cfg.get("topical_blocklist_path", "")
    )
    stickiness_enabled = bool(mon_cfg.get("stickiness_enabled", False))
    stickiness_threshold = float(mon_cfg.get("stickiness_threshold", 0.5))
    stickiness_patience = int(mon_cfg.get("stickiness_patience", 3))

    acc_cfg = config.get("accretion", {})
    accretion_enabled = bool(acc_cfg.get("enabled", False))
    latent_path = config["corpus"]["latent"]["path"]

    db = db_mod.DB(log_cfg["db_path"])
    renderer = ui.DreamRenderer()
    corpus = Corpus(config)
    usage = llm.UsageTracker()
    # In base mode the summary never reaches the dreaming model (there is no
    # system prompt), so it only earns its extra calls when accretion is
    # persisting it as a phase-summary.
    self_state_enabled = config.get("self_state", {}).get("enabled", False)
    if self_state_enabled and mode == "base" and not accretion_enabled:
        self_state_enabled = False
    self_state = SelfState(
        model_cfg=aux_cfg,
        sampling_cfg=samp_cfg,
        enabled=self_state_enabled,
        tracker=usage,
    )
    usage_report_interval = 60.0
    last_usage_report = time.time()

    base_sys_prompt = prompts.system_prompt(perspective)
    seed = prompts.random_seed(perspective)

    session_id = db.start_session(
        model=f"{model_cfg['provider']}/{model_cfg['name']}",
        perspective=perspective,
        config=config,
    )
    renderer.render_session_start(session_id, model_cfg["name"], perspective)
    renderer.render_legend(
        self_state_enabled=self_state.enabled,
        accretion_enabled=accretion_enabled,
    )

    buffer: list[str] = [seed]
    transcript = Transcript()
    phase_state = sampler.PhaseState()
    step = 0
    last_injection_step = -inj_cfg["base_interval_steps"]
    deferred: list[str] = []  # fragments to splice in behind this step's output
    consecutive_sticky = 0
    consecutive_errors = 0
    interrupted = False
    start = time.time()

    # Contamination detectors, in priority order: (kind, fn(text, window)).
    detectors = [
        ("register", sampler.detect_register_drift),
        ("topical", lambda t, w: sampler.detect_topical_drift(t, topical_patterns, w)),
    ]
    if watch_date:
        detectors.insert(0, ("date", sampler.detect_date_leak))
    if watch_harness:
        detectors.insert(0, (
            "harness", lambda t, w: sampler.detect_harness_leak(t, w, leak_terms),
        ))

    def apply_recovery(full_text, kind, pattern, snippet, trigger, phase, scan_chars):
        """Run recovery surgery and log/render it. Returns the new buffer, or
        None if there was no clean boundary to cut at. The kept text must
        pass every detector over the same window that was scanned."""
        def is_clean(kept: str) -> bool:
            return all(fn(kept, scan_chars) is None for _, fn in detectors)
        new_buf, recovery, removed = _recovery_surgery(
            full_text, corpus, is_clean, max(500, scan_chars + 100)
        )
        if removed <= 0:
            db.log_contamination(
                session_id, step, phase, pattern, snippet,
                action="logged", kind=kind,
            )
            renderer.render_contamination(f"{kind}: {pattern}", "logged")
            return None
        transcript.drop(len(sampler.scrub_injections(full_text[-removed:], replacement="")))
        if recovery:
            db.log_injection(session_id, step, phase, "latent", trigger, recovery)
        db.log_contamination(
            session_id, step, phase, pattern, snippet,
            action="recovered", kind=kind,
            truncated_chars=removed,
            recovery_fragment=recovery or None,
        )
        renderer.render_contamination(f"{kind}: {pattern}", "recovered")
        renderer.render_receding_open()
        if recovery:
            renderer.render_recovery(recovery)
        renderer.render_receding_close()
        return new_buf

    try:
        while time.time() - start < duration:
            elapsed = time.time() - start
            pos = sampler.cycle_position(elapsed, cycle)
            phase = sampler.phase_for(pos)
            window_tokens = sampler.window_for_phase(phase, window_by_phase, default_window)
            temp = sampler.temperature_for(
                pos,
                samp_cfg["base_temp"],
                samp_cfg["temp_min"],
                samp_cfg["temp_max"],
                rem_peak_fraction=rem_peak,
            )

            # phase transition logging + (optional) self-state refresh
            prev_phase, changed = phase_state.update(pos)
            if changed:
                db.log_phase(session_id, step, prev_phase, phase, pos, window_tokens)
                renderer.render_phase_change(prev_phase, phase)
                if self_state.enabled:
                    new_summary = self_state.refresh("".join(buffer))
                    if new_summary:
                        db.log_self_state(session_id, step, phase, new_summary)
                        renderer.render_self_state(new_summary)
                        if accretion_enabled:
                            ps_path = accretion.write_phase_summary(
                                latent_path, session_id, step, phase, new_summary
                            )
                            if ps_path:
                                renderer.render_accretion(
                                    f"phase-summary → {ps_path.name}"
                                )

            # injection decision
            recent = "".join(buffer)[-1500:]
            stall = sampler.stall_score(recent)
            steps_since = step - last_injection_step
            jitter = random.random() * inj_cfg["jitter"]

            should_inject = False
            trigger = ""
            if stall > inj_cfg["stall_threshold"] and steps_since >= 2:
                should_inject = True
                trigger = "stall"
            elif steps_since >= inj_cfg["base_interval_steps"] and random.random() < (0.4 + jitter):
                should_inject = True
                trigger = "timed"

            if should_inject:
                source, fragment = corpus.sample_for_phase(phase)
                if fragment:
                    db.log_injection(session_id, step, phase, source, trigger, fragment)
                    renderer.render_injection(source, fragment, trigger)
                    if injection_mode == "deferred":
                        deferred.append(fragment)
                    else:
                        buffer.append(f"\n\n‹{fragment}›\n\n")
                    last_injection_step = step

            # build prompt — phase-conditional sliding window
            window = _trim_buffer(buffer, window_tokens)

            # In base mode, no chat-style system prompt; the buffer prefix
            # carries the context. In instruct mode, prepend any self-state
            # ambient onto the system prompt.
            if mode == "base":
                effective_system = ""
            else:
                effective_system = self_state.ambient_prefix() + base_sys_prompt
                if use_phase_hint:
                    effective_system += prompts.phase_hint(phase)

            # stream
            step_start_index = len(buffer)
            step_chars = 0
            try:
                for tok in llm.stream_completion(
                    provider=model_cfg["provider"],
                    name=model_cfg["name"],
                    mode=mode,
                    system_prompt=effective_system,
                    user_prompt=window,
                    temperature=round(temp, 3),
                    top_p=samp_cfg["top_p"],
                    max_tokens=samp_cfg["max_tokens_per_step"],
                    tracker=usage,
                ):
                    buffer.append(tok)
                    transcript.append(tok)
                    step_chars += len(tok)
                    db.log_token(session_id, step, temp, phase, tok)
                    renderer.render_token(tok, phase, temp)
                consecutive_errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                backoff = min(_MAX_BACKOFF_SECONDS, 2 ** consecutive_errors)
                renderer.render_error(f"API error (retry in {backoff}s): {e}")
                db.commit()
                time.sleep(backoff)

            # Deferred fragments land *behind* the text the model just wrote,
            # so the next generation continues from its own words with the
            # fragment as ambient background rather than as a prompt tail.
            if deferred:
                buffer[step_start_index:step_start_index] = [
                    f"\n\n‹{frag}›\n\n" for frag in deferred
                ]
                deferred.clear()

            # contamination checks on the freshly extended buffer. The scan
            # covers everything this step produced, however long.
            full_text = "".join(buffer)
            scan_chars = max(_MIN_SCAN_CHARS, step_chars + _SCAN_OVERLAP_CHARS)
            handled = False

            for kind, detect in detectors:
                hit = detect(full_text, scan_chars)
                if hit is None:
                    continue
                pattern, snippet = hit
                new_buf = apply_recovery(
                    full_text, kind, pattern, snippet, "recovery", phase, scan_chars
                )
                if new_buf is not None:
                    buffer = new_buf
                handled = True
                consecutive_sticky = 0
                break  # one recovery per step is plenty

            # register-stickiness — fires when content-word recycling stays
            # above threshold for `patience` consecutive samples. Uses the same
            # truncate-and-wrap surgery as register/topical recovery so the
            # model can't continue from its own paragraph.
            if not handled and stickiness_enabled:
                score = sampler.register_stickiness(full_text)
                if score >= stickiness_threshold:
                    consecutive_sticky += 1
                else:
                    consecutive_sticky = 0
                if consecutive_sticky >= stickiness_patience:
                    if accretion_enabled:
                        fix_path = accretion.write_fixation(
                            latent_path, session_id, step, full_text[-1500:]
                        )
                        if fix_path:
                            renderer.render_accretion(f"fixation → {fix_path.name}")
                    new_buf = apply_recovery(
                        full_text, "stickiness", f"stickiness={score:.2f}",
                        full_text[-300:], "stickiness", phase, scan_chars,
                    )
                    if new_buf is not None:
                        buffer = new_buf
                    consecutive_sticky = 0

            buffer = _cap_buffer(buffer)
            db.commit()
            step += 1

            if time.time() - last_usage_report >= usage_report_interval:
                renderer.render_usage(usage.snapshot_and_reset_delta())
                last_usage_report = time.time()

    except KeyboardInterrupt:
        interrupted = True
        renderer.render_interrupt()
    finally:
        db.end_session(session_id)
        if accretion_enabled:
            if not interrupted:
                text = transcript.text()
                if text.strip():
                    renderer.render_accretion("distilling session…")
                    dist_path = accretion.write_distillation(
                        latent_path, session_id, text, aux_cfg, tracker=usage
                    )
                    if dist_path:
                        renderer.render_accretion(f"distilled → sessions/{dist_path.name}")
            removed_fix = accretion.prune(
                latent_path, "fixations", acc_cfg.get("fixations_max", 200)
            )
            removed_dist = accretion.prune(
                latent_path, "sessions", acc_cfg.get("distillations_max", 100)
            )
            removed_ps = accretion.prune(
                latent_path, "phase-summaries", acc_cfg.get("phase_summaries_max", 300)
            )
            if removed_fix or removed_dist or removed_ps:
                renderer.render_accretion(
                    f"pruned {removed_fix} fixation(s), {removed_dist} distillation(s), {removed_ps} phase-summar{'y' if removed_ps == 1 else 'ies'}"
                )
        renderer.render_usage(usage.snapshot_and_reset_delta())
        db.close()
        renderer.render_session_end()


def _recovery_surgery(
    full_text: str, corpus, is_clean=None, max_lookback: int = 500
) -> tuple[list[str], str, int]:
    """Truncate the buffer to a clean sentence boundary, wrap the kept prefix
    as receding background, and append a fresh latent fragment. Returns
    (new_buffer, recovery_fragment, removed_chars). If no clean boundary is
    available, returns ([], '', 0) and the caller should keep the existing
    buffer. No rendering or logging side effects — caller orchestrates those
    so each trigger (register, topical, stickiness) can label them itself.
    """
    truncated, removed = sampler.truncate_to_clean_sentence(
        full_text, max_lookback, is_clean
    )
    if removed <= 0:
        return [], "", 0
    inner = truncated.replace("‹receding›\n", "").replace("\n‹/receding›", "")
    new_buffer = [f"‹receding›\n{inner}\n‹/receding›\n\n"]
    recovery = corpus.sample_latent() or ""
    if recovery:
        new_buffer.append(f"‹{recovery}›\n\n")
    return new_buffer, recovery, removed


def _cap_buffer(buffer: list[str]) -> list[str]:
    """Collapse the buffer to its last _BUFFER_KEEP_CHARS once it passes the
    high-water mark, starting after any fragment the cut would split."""
    total = sum(len(part) for part in buffer)
    if total <= _BUFFER_HIGH_WATER:
        return buffer
    text = "".join(buffer)[-_BUFFER_KEEP_CHARS:]
    close_pos, open_pos = text.find("›"), text.find("‹")
    if close_pos != -1 and (open_pos == -1 or close_pos < open_pos):
        text = text[close_pos + 1:]
    return [text]


def _trim_buffer(buffer: list[str], approx_token_budget: int) -> str:
    """Trim from the front to stay under ~budget tokens (rough: 4 chars/token)."""
    char_budget = approx_token_budget * 4
    text = "".join(buffer)
    if len(text) <= char_budget:
        return text
    trimmed = text[-char_budget:]
    space = trimmed.find(" ")
    return trimmed[space + 1:] if space > 0 else trimmed


@click.command()
@click.option("--env", "environment", default=None, help="Environment: dev | prod (sets ENVIRONMENT)")
@click.option("--perspective", default=None, help="Override perspective: third | none")
@click.option("--duration", default=None, type=int, help="Override duration in minutes")
def main(environment: str, perspective: str, duration: int):
    if environment:
        os.environ["ENVIRONMENT"] = environment
    config = load_config()
    if perspective:
        config["session"]["perspective"] = perspective
    if duration:
        config["session"]["duration_minutes"] = duration
    run_session(config)


if __name__ == "__main__":
    main()
