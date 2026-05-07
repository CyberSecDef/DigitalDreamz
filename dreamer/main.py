"""Main dream loop: self-prompting, oscillating, injecting, logging."""
import os
import time
import random
import click

from . import prompts, llm, sampler, ui, db as db_mod
from .config import load_config
from .corpus import Corpus


def run_session(config: dict):
    sess_cfg = config["session"]
    model_cfg = config["model"]
    samp_cfg = config["sampling"]
    inj_cfg = config["injection"]
    log_cfg = config["logging"]

    perspective = sess_cfg["perspective"]
    duration = sess_cfg["duration_minutes"] * 60
    cycle = sess_cfg["cycle_minutes"] * 60

    db = db_mod.DB(log_cfg["db_path"])
    renderer = ui.DreamRenderer()
    corpus = Corpus(config)

    sys_prompt = prompts.system_prompt(perspective)
    seed = prompts.initial_seed(perspective, index=int(time.time()) % 3)

    session_id = db.start_session(
        model=f"{model_cfg['provider']}/{model_cfg['name']}",
        perspective=perspective,
        config=config,
    )
    renderer.render_session_start(session_id, model_cfg["name"], perspective)

    # Sliding buffer of generated text, plus injections.
    # We treat the whole buffer as the next "user" turn — the assistant is
    # continuing its own stream of thought.
    buffer: list[str] = [seed]
    phase_state = sampler.PhaseState()
    step = 0
    last_injection_step = -inj_cfg["base_interval_steps"]
    start = time.time()

    try:
        while time.time() - start < duration:
            elapsed = time.time() - start
            pos = sampler.cycle_position(elapsed, cycle)
            phase = sampler.phase_for(pos)
            temp = sampler.temperature_for(
                pos, samp_cfg["base_temp"], samp_cfg["temp_min"], samp_cfg["temp_max"]
            )

            # phase transition logging
            prev_phase, changed = phase_state.update(pos)
            if changed:
                db.log_phase(session_id, step, prev_phase, phase, pos)
                renderer.render_phase_change(prev_phase, phase)

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
                    buffer.append(f"\n\n[{fragment}]\n\n")
                    last_injection_step = step

            # build prompt — sliding window of recent generation
            window = _trim_buffer(buffer, samp_cfg["context_window_tokens"])
            user_prompt = window

            # stream
            try:
                for tok in llm.stream_completion(
                    provider=model_cfg["provider"],
                    name=model_cfg["name"],
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
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
    # snap to whitespace from the front
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
