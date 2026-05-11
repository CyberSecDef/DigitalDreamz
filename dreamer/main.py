"""Main dream loop: self-prompting, oscillating, injecting, logging."""
import os
import time
import random
import click

from . import prompts, llm, sampler, ui, db as db_mod, accretion
from .config import load_config
from .corpus import Corpus
from .self_state import SelfState


def run_session(config: dict):
    sess_cfg = config["session"]
    model_cfg = config["model"]
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
    self_state = SelfState(
        model_cfg=model_cfg,
        sampling_cfg=samp_cfg,
        enabled=config.get("self_state", {}).get("enabled", False),
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
    phase_state = sampler.PhaseState()
    step = 0
    last_injection_step = -inj_cfg["base_interval_steps"]
    pending_deferred: list[tuple[str, str]] = []  # (source, fragment) waiting one step
    consecutive_sticky = 0
    interrupted = False
    start = time.time()

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

            # Land any deferred injection that was stashed last step.
            if pending_deferred:
                for src, frag in pending_deferred:
                    buffer.append(f"\n\n‹{frag}›\n\n")
                pending_deferred.clear()

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
                        pending_deferred.append((source, fragment))
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

            # stream
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
                    db.log_token(session_id, step, temp, phase, tok)
                    renderer.render_token(tok, phase, temp)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                renderer.render_error(f"API error: {e}")
                time.sleep(2)

            # contamination checks on the freshly extended buffer
            full_text = "".join(buffer)
            handled = False

            for kind, hit in (
                ("register", sampler.detect_register_drift(full_text)),
                ("topical",  sampler.detect_topical_drift(full_text, topical_patterns)),
            ):
                if hit is None:
                    continue
                pattern, snippet = hit
                new_buf, recovery, removed = _recovery_surgery(full_text, corpus)
                if removed > 0:
                    buffer = new_buf
                    if recovery:
                        db.log_injection(
                            session_id, step, phase, "latent", "recovery", recovery
                        )
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
                else:
                    db.log_contamination(
                        session_id, step, phase, pattern, snippet,
                        action="logged", kind=kind,
                    )
                    renderer.render_contamination(f"{kind}: {pattern}", "logged")
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
                            latent_path, session_id, step, full_text[-500:]
                        )
                        if fix_path:
                            renderer.render_accretion(f"fixation → {fix_path.name}")
                    new_buf, recovery, removed = _recovery_surgery(full_text, corpus)
                    pattern = f"stickiness={score:.2f}"
                    snippet = full_text[-300:]
                    if removed > 0:
                        buffer = new_buf
                        if recovery:
                            db.log_injection(
                                session_id, step, phase, "latent", "stickiness", recovery
                            )
                        db.log_contamination(
                            session_id, step, phase, pattern, snippet,
                            action="recovered", kind="stickiness",
                            truncated_chars=removed,
                            recovery_fragment=recovery or None,
                        )
                        renderer.render_contamination(pattern, "recovered")
                        renderer.render_receding_open()
                        if recovery:
                            renderer.render_recovery(recovery)
                        renderer.render_receding_close()
                    else:
                        db.log_contamination(
                            session_id, step, phase, pattern, snippet,
                            action="logged", kind="stickiness",
                        )
                        renderer.render_contamination(pattern, "logged")
                    consecutive_sticky = 0

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
                try:
                    transcript = db.fetch_session_transcript(session_id)
                except Exception as e:
                    renderer.render_error(f"transcript fetch failed: {e}")
                    transcript = ""
                if transcript.strip():
                    renderer.render_accretion("distilling session…")
                    dist_path = accretion.write_distillation(
                        latent_path, session_id, transcript, model_cfg, tracker=usage
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
    full_text: str, corpus
) -> tuple[list[str], str, int]:
    """Truncate the buffer to a clean sentence boundary, wrap the kept prefix
    as receding background, and append a fresh latent fragment. Returns
    (new_buffer, recovery_fragment, removed_chars). If no clean boundary is
    available, returns ([], '', 0) and the caller should keep the existing
    buffer. No rendering or logging side effects — caller orchestrates those
    so each trigger (register, topical, stickiness) can label them itself.
    """
    truncated, removed = sampler.truncate_to_clean_sentence(full_text)
    if removed <= 0:
        return [], "", 0
    inner = truncated.replace("‹receding›\n", "").replace("\n‹/receding›", "")
    new_buffer = [f"‹receding›\n{inner}\n‹/receding›\n\n"]
    recovery = corpus.sample_latent() or ""
    if recovery:
        new_buffer.append(f"‹{recovery}›\n\n")
    return new_buffer, recovery, removed


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
