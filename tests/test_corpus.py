import time

from dreamer.corpus import LatentCorpus, WorldEvents


def test_latent_samples_whole_paragraphs(tmp_path):
    paras = [
        "The room had one window. The window faced a garden.",
        "There was a house with seven rooms. Whoever entered the seventh found the first.",
    ]
    (tmp_path / "rooms.txt").write_text("\n\n".join(paras))
    lc = LatentCorpus(str(tmp_path), chunk_chars=280)
    for _ in range(30):
        assert lc.sample() in paras


def test_latent_long_paragraph_yields_whole_sentences(tmp_path):
    sentences = [f"Sentence number {i} drifts past the window." for i in range(40)]
    (tmp_path / "long.txt").write_text(" ".join(sentences))
    lc = LatentCorpus(str(tmp_path), chunk_chars=120)
    for _ in range(30):
        chunk = lc.sample()
        assert len(chunk) <= 121
        assert chunk.startswith("Sentence number") and chunk.endswith(".")


def test_latent_line_files_split_per_line(tmp_path):
    lines = [f"fragment {i}: a kettle in the wrong year" for i in range(20)]
    (tmp_path / "sessions.md").write_text("\n".join(lines))
    lc = LatentCorpus(str(tmp_path), chunk_chars=100)
    for _ in range(20):
        assert lc.sample() in lines


def test_latent_strips_brackets(tmp_path):
    (tmp_path / "f.txt").write_text("residue ‹receding› of an old fixation›")
    lc = LatentCorpus(str(tmp_path))
    out = lc.sample()
    assert "‹" not in out and "›" not in out


def test_latent_weights_exclude(tmp_path):
    (tmp_path / "README.md").write_text("docs")
    (tmp_path / "a.txt").write_text("the only fragment")
    (tmp_path / "weights.txt").write_text("README.md 0  # docs\n")
    lc = LatentCorpus(str(tmp_path))
    assert lc.sample() == "the only fragment"


def test_world_failed_refresh_is_not_retried_every_sample(monkeypatch):
    calls = []

    def boom(self, url):
        calls.append(url)
        raise OSError("offline")

    monkeypatch.setattr(WorldEvents, "_fetch", boom)
    we = WorldEvents(["http://a", "http://b"])
    assert calls == ["http://a", "http://b"]
    we.fragments = ["a headline"]
    for _ in range(10):
        we.sample()
    time.sleep(0.05)
    assert len(calls) == 2
