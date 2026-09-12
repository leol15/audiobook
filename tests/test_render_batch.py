"""Render batching: pending chunks are grouped by voice and sent in batches."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import soundfile as sf

from ab.config import BookConfig, BookPaths
from ab.lines import read_lines, write_attribution
from ab.models import Line
from ab.stages import s05_render as render
from ab.tts.tone import ToneBackend


class BatchTone(ToneBackend):
    """Tone backend that also accepts batches and records what it was sent."""

    name = "batchtone-test"
    max_chars = 30
    batch_size = 4
    languages: ClassVar[set[str]] = {"en", "zh"}

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[tuple[str, int]] = []  # (voice, items) per model call

    def synthesize(self, text, voice, *, lang, **params):
        self.calls.append((voice, 1))
        return super().synthesize(text, voice, lang=lang, **params)

    def synthesize_batch(self, texts, voice, *, lang, **params):
        self.calls.append((voice, len(texts)))
        return [super().synthesize(t, voice, lang=lang, **params) for t in texts]


def _book(tmp_path, batch=None):
    root = tmp_path / "book"
    root.mkdir(parents=True)
    cfg = BookConfig(title="t", language="en", tts={"backend": "tone", "batch": batch},
                     voices={"narrator": "a220", "Ann": "a440", "_default": "a330"})
    paths = BookPaths(root)
    paths.work.mkdir()
    long = "This sentence is long. " * 3  # > max_chars -> several chunks
    lines = [
        Line(id="c0p0s0", chapter=0, para=0, kind="narration", speaker="narrator", text="One."),
        Line(id="c0p1s0", chapter=0, para=1, kind="dialogue", speaker="Ann", text="Two words."),
        Line(id="c0p2s0", chapter=0, para=2, kind="narration", speaker="narrator", text=long.strip()),
        Line(id="c0p3s0", chapter=0, para=3, kind="dialogue", speaker="Ann", text="Three."),
        Line(id="c0p4s0", chapter=0, para=4, kind="narration", speaker="narrator", text="Four four."),
        Line(id="c0p5s0", chapter=0, para=5, kind="dialogue", speaker="Bob", text="Five."),
    ]
    write_attribution(paths, 0, lines)
    return paths, cfg


def test_batches_group_by_voice_and_cap_size(tmp_path):
    paths, cfg = _book(tmp_path)
    backend = BatchTone()
    render._run(paths, cfg, backend, force=False, chapters=None)
    by_voice = {}
    for voice, n in backend.calls:
        by_voice.setdefault(voice, []).append(n)
    # narrator: 1 + 3 chunks + 1 = 5 items -> a batch of 4 and a leftover single
    assert sorted(by_voice["a220"]) == [1, 4]
    assert by_voice["a440"] == [2]
    assert by_voice["a330"] == [1]
    lines = read_lines(paths)
    assert all(ln.audio and (paths.work / ln.audio).exists() for ln in lines)
    assert all(ln.backend == "batchtone-test" for ln in lines)
    # The multi-chunk line is assembled from all its chunks, in order.
    long_audio, _ = sf.read(paths.work / lines[2].audio)
    short_audio, _ = sf.read(paths.work / lines[0].audio)
    assert len(long_audio) > 3 * len(short_audio)


def test_batch_config_overrides_backend_default(tmp_path):
    paths, cfg = _book(tmp_path, batch=2)
    backend = BatchTone()
    render._run(paths, cfg, backend, force=False, chapters=None)
    assert max(n for _, n in backend.calls) == 2


def test_batch_render_matches_single_render(tmp_path):
    paths, cfg = _book(tmp_path)
    render._run(paths, cfg, BatchTone(), force=False, chapters=None)
    batched = {ln.id: sf.read(paths.work / ln.audio)[0] for ln in read_lines(paths)}
    paths2, cfg2 = _book(tmp_path / "again", batch=1)
    render._run(paths2, cfg2, BatchTone(), force=False, chapters=None)
    single = {ln.id: sf.read(paths2.work / ln.audio)[0] for ln in read_lines(paths2)}
    for k, audio in batched.items():
        assert np.array_equal(audio, single[k])


def test_plain_backend_stays_single(tmp_path):
    paths, cfg = _book(tmp_path)
    calls = []
    tone = ToneBackend()
    tone.max_chars = 30
    orig = tone.synthesize

    def spy(text, voice, *, lang, **p):
        calls.append(text)
        return orig(text, voice, lang=lang, **p)

    tone.synthesize = spy
    render._run(paths, cfg, tone, force=False, chapters=None)
    assert len(calls) == 8  # 5 single-chunk lines + 3 chunks
    assert all(json.loads(l)["audio"] for l in paths.chapter_render(0).read_text().splitlines())
