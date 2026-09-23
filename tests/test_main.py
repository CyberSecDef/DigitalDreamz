from dreamer.main import Transcript, _cap_buffer, _BUFFER_KEEP_CHARS


def test_transcript_drop():
    t = Transcript()
    for tok in ["Fog ", "in the hall. ", "Let me explain"]:
        t.append(tok)
    t.drop(len("Let me explain"))
    assert t.text() == "Fog in the hall. "


def test_cap_buffer_skips_split_fragment():
    head = "x" * 20000
    buf = [head, "‹" + "f" * 100 + "›", "y" * (_BUFFER_KEEP_CHARS - 50)]
    out = _cap_buffer(buf)
    assert len(out) == 1
    assert "›" not in out[0] and out[0].startswith("y")


def test_cap_buffer_noop_under_high_water():
    buf = ["a", "b"]
    assert _cap_buffer(buf) is buf
