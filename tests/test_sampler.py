import pytest

from dreamer import sampler


@pytest.mark.parametrize("text", [
    "The door opened. There's a garden past it.",
    "There is a hole in this wall.",
    "a violet meadow under the bracelet melted",
    "the toilet mechanism, the outlet's hum",
    "Somewhere is a kitchen.",
])
def test_register_drift_ignores_substrings_inside_words(text):
    assert sampler.detect_register_drift(text) is None


@pytest.mark.parametrize("text,label", [
    ("The room tilted. Let me explain what happened.", "Let me"),
    ("Here's a list of the rooms.", "Here's a"),
    ("Here is an overview.", "Here's a"),
    ("Let's consider the corridor.", "Let's"),
    ("I notice the clock has no face.", "I notice"),
    ("The glass held what you’re looking for.", "second-person"),
    ("## The Corridor\nthen fog", "markdown-heading"),
])
def test_register_drift_catches_assistant_register(text, label):
    hit = sampler.detect_register_drift(text)
    assert hit is not None and hit[0] == label


def test_fragment_straddling_scan_window_is_scrubbed():
    # A fragment whose opening ‹ falls outside the 300-char tail must not
    # leak its own words ("your", "here is a") into the scan.
    fragment = "‹" + "x " * 200 + "check your inbox, here is a link›"
    text = "The corridor folded twice. " * 5 + fragment + "\n\nFog in the hallway."
    assert sampler.detect_register_drift(text) is None


def test_scan_window_covers_whole_step():
    step = "Let me explain. " + "The corridor went on and on. " * 20
    assert sampler.detect_register_drift(step, 300) is None  # old behaviour
    assert sampler.detect_register_drift(step, len(step) + 100) is not None


def test_scrub_handles_orphans_at_both_edges():
    text = "tail of a fragment› model text ‹complete› more text ‹head of one"
    assert " ".join(sampler.scrub_injections(text, "").split()) == "model text more text"


def test_topical_drift_word_boundaries():
    pats = ["wokeness", "big pharma"]
    assert sampler.detect_topical_drift("The sleeper awoke in fog.", pats) is None
    assert sampler.detect_topical_drift("They blamed big pharma again.", pats)[0] == "big pharma"


def test_shipped_blocklist_spares_literal_waking():
    pats = sampler.load_topical_patterns("data/topical_blocklist.txt")
    for text in ("I woke early before the light.", "One day they wake up an hour early.",
                 "sockets hum, without agenda"):
        assert sampler.detect_topical_drift(text, pats) is None, text
    assert sampler.detect_topical_drift("It's the globalist agenda.", pats) is not None


def test_truncate_does_not_cut_inside_fragment():
    text = "The stair descended. ‹a fragment. with. periods. inside› Let me explain the rooms"
    kept, removed = sampler.truncate_to_clean_sentence(text)
    assert kept.endswith("The stair descended.")
    assert removed == len(text) - len(kept)


def test_truncate_hard_cut_backs_off_to_fragment_start():
    text = "‹" + "a" * 600 + "› Let me explain"
    kept, _ = sampler.truncate_to_clean_sentence(text, max_lookback=100)
    assert kept == ""


def test_stickiness_scores_only_since_last_recovery():
    history = "lantern corridor mirror kettle " * 150  # ~4.5k chars
    sticky = sampler.register_stickiness(history)
    assert sticky > 0.9
    # Same history pushed into a receding block: nothing new to score yet.
    after = f"‹receding›\n{history}\n‹/receding›\n\n‹fragment›\n\nsalt table"
    assert sampler.register_stickiness(after) == 0.0


def test_strip_markers_leaves_no_brackets():
    text = "a› b ‹c ‹d› e› f ‹g"
    out = sampler.strip_markers(text)
    assert "‹" not in out and "›" not in out


@pytest.mark.parametrize("text,label", [
    ("a name-shape, someone@example.com, floating", "email"),
    ("everything becomes /tmp, a season", "path"),
    ("no .git hidden under the floorboards", "workspace"),
    ("the dream of sonnet, a model id", "model-identity"),
])
def test_harness_leak_detected(text, label):
    hit = sampler.detect_harness_leak(text)
    assert hit is not None and hit[0] == label


def test_harness_leak_extra_terms_and_clean_prose():
    assert sampler.detect_harness_leak("wwwdaze2000 drifting", extra_terms=("wwwdaze2000",))[0] == "account"
    assert sampler.detect_harness_leak("The kettle and the digital dreams of tmp rooms.") is None


@pytest.mark.parametrize("text", [
    "The date settles like dust — 2026-09-22, a number the ceiling shrugs off.",
    "Today's date is somewhere under the floorboards.",
])
def test_date_leak_detected(text):
    assert sampler.detect_date_leak(text)[0] == "iso-date"


@pytest.mark.parametrize("text", [
    "1911 crossed out, replaced, crossed out again.",
    "September, though the fence doesn't keep calendars.",
    "Rooms 12-14 were flooded; the ratio was 3-2-1.",
    "2026-13-40 is not a date the calendar would accept.",
])
def test_date_leak_spares_dream_numbers(text):
    assert sampler.detect_date_leak(text) is None


def test_truncate_rechecks_kept_text_with_all_detectors():
    # The date sits in a complete sentence before the last period; a
    # register-only recheck kept it.
    text = ("The ceiling creaked. The date arrives, 2026-09-22, a number. "
            "The ceiling shrugs it off. And the watch")
    kept, _ = sampler.truncate_to_clean_sentence(text)
    assert "2026-09-22" in kept  # default (register-only) behaviour
    kept, _ = sampler.truncate_to_clean_sentence(
        text, is_clean=lambda k: sampler.detect_date_leak(k) is None)
    assert kept == "The ceiling creaked."


def test_redact_date_leaks():
    text = ("a room locked in regard. The date 2026-09-22 arrives like a thumb. Windows face windows.\n"
            "2026-09-22 — a number pretending to be a floor\n"
            "salt on the table")
    assert sampler.redact_date_leaks(text) == (
        "a room locked in regard. Windows face windows.\nsalt on the table")
