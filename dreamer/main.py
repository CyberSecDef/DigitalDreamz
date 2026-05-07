"""Main dream loop: self-prompting, oscillating, injecting, logging."""
import os
import time
import random
import click

from . import prompts, llm, sampler, ui, db as db_mod
from .config import load_config
from .corpus import Corpus
from .self_state import SelfState


def run_session(config: dict):
    sess_cfg = config["session"]
    model_cfg = config["model"]
    samp_cfg = config["sampling"]
    inj_cfg = config["injection"]
    log_cfg = config["logging"]

    perspective = sess_cfg["perspective"]
    duration = sess_cfg["duration_minutes"] * 60
    cycle = sess_cfg["cycle_minutes"] * 60
    mode = model_cfg.get("mode", "instruct")
    injection_mode = inj_cfg.get("mode", "visible")
    window_by_phase = samp_cfg.get("context_window_by_phase", {})
    default_window = samp_cfg["context_window_tokens"]

    db = db_mod.DB(log_cfg["db_path"])
    renderer = ui.DreamRenderer()
    corpus = Corpus(config)
    self_state = SelfState(
        model_cfg=model_cfg,
        sampling_cfg=samp_cfg,
        enabled=config.get("self_state", {}).get("enabled", False),
    )

    base_sys_prompt = prompts.system_prompt(perspective)
    seed = prompts.initial_seed(perspective, index=int(time.time()) % 3)

    session_id = db.start_session(
        model=f"{model_cfg['provider']}/{model_cfg['name']}",
        perspective=perspective,
        config=config,
    )
    renderer.render_session_start(session_id, model_cfg["name"], perspective)

    buffer: list[str] = [seed]
    phase_state = sampler.PhaseState()
    step = 0
    last_injection_step = -inj_cfg["base_interval_steps"]
    pending_deferred: list[tuple[str, str]] = []  # (source, fragment) waiting one step
    start = time.time()

    try:
        while time.time() - start < duration:
            elapsed = time.time() - start
            pos = sampler.cycle_position(elapsed, cycle)
            phase = sampler.phase_for(pos)
            window_tokens = sampler.window_for_phase(phase, window_by_phase, default_window)
            temp = sampler.temperature_for(
                pos, samp_cfg["base_temp"], samp_cfg["temp_min"], samp_cfg["temp_max"]
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

            # Land any deferred injection that was stashed last step.
            if pending_deferred:
                for src, frag in pending_deferred:
                    buffer.append(f"\n\n[{frag}]\n\n")
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
                        buffer.append(f"\n\n[{fragment}]\n\n")
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
                ):
                    buffer.append(tok)
                    db.log_token(session_id, step, temp, phase, tok)
                    renderer.render_token(tok, phase, temp)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                renderer.console.print(f"\n[red]API error: {e}[/red]")
                time.sleep(2)

            # register-drift check on the freshly extended buffer
            full_text = "".join(buffer)
            drift = sampler.detect_register_drift(full_text)
            if drift is not None:
                pattern, snippet = drift
                truncated, removed = sampler.truncate_to_clean_sentence(full_text)
                if removed > 0:
                    buffer = [truncated]
                    recovery = corpus.sample_latent() or ""
                    if recovery:
                        buffer.append(f"\n\n[{recovery}]\n\n")
                        db.log_injection(
                            session_id, step, phase, "latent", "recovery", recovery
                        )
                    db.log_contamination(
                        session_id, step, phase, pattern, snippet,
                        action="recovered",
                        truncated_chars=removed,
                        recovery_fragment=recovery or None,
                    )
                    renderer.render_contamination(pattern, "recovered")
                else:
                    db.log_contamination(
                        session_id, step, phase, pattern, snippet,
                        action="logged",
                    )
                    renderer.render_contamination(pattern, "logged")

            step += 1

    except KeyboardInterrupt:
        renderer.console.print("\n[dim]interrupted[/dim]")
    finally:
        db.end_session(session_id)
        db.close()
        renderer.render_session_end()


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
