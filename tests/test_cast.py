"""Cast stage at scale: per-chapter discovery (cached), incremental merge, ranking, main cap."""

from __future__ import annotations

from ab.config import BookConfig, BookPaths, Cast, Character
from ab.models import Chapter, ChapterList
from ab.stages import s03_cast as cast_stage

CHAPTERS = [
    Chapter(index=0, title="One", paragraphs=[
        '"Good morning," said Mr. Darcy.', '"Indeed," Elizabeth replied.', '"Quite," said Mr. Darcy.']),
    Chapter(index=1, title="Two", paragraphs=[
        '"Tea?" asked Jane.', '"Please," said Lizzy.']),
    Chapter(index=2, title="Three", paragraphs=[
        '"Sir," said the butler.', '"Go," said Darcy.', '"Now," said Darcy.']),
]


class ScriptedLLM:
    """Answers discover prompts from a table and merge prompts with a fixed policy."""

    def __init__(self, discover: dict[str, list[dict]]):
        self.discover = discover
        self.calls: list[str] = []
        self.num_ctx = 16384

    def window_budget(self, overhead: str) -> int:
        return 4000

    def json(self, prompt: str, schema: dict, system: str = "") -> dict:
        self.calls.append(prompt)
        if "SPEAKS" in prompt:
            for key, chars in self.discover.items():
                if key in prompt:
                    return {"characters": chars}
            return {"characters": []}
        # merge: anything called Darcy is the known Mr. Darcy; Lizzy is Elizabeth
        groups = []
        for line in prompt.split("New character entries")[1].splitlines():
            if not line.startswith("- "):
                continue
            name = line[2:].split(" (aka")[0]
            if name == "Darcy":
                groups.append({"canonical": "Mr. Darcy", "aliases": ["Darcy"], "description": ""})
            elif name == "Lizzy":
                groups.append({"canonical": "Elizabeth", "aliases": ["Lizzy"], "description": ""})
            else:
                groups.append({"canonical": name, "aliases": [], "description": f"{name} desc"})
        return {"groups": groups}

    def unload(self):
        pass


DISCOVER = {
    "Good morning": [{"name": "Mr. Darcy", "aliases": [], "description": "proud"},
                     {"name": "Elizabeth", "aliases": [], "description": "witty"}],
    "Tea?": [{"name": "Jane", "aliases": [], "description": "kind"},
             {"name": "Lizzy", "aliases": [], "description": ""}],
    "butler": [{"name": "the butler", "aliases": [], "description": "servant"},
               {"name": "Darcy", "aliases": [], "description": ""}],
}


def _book(tmp_path, **cast_cfg):
    paths = BookPaths(tmp_path)
    paths.work.mkdir()
    paths.chapters_norm.write_text(ChapterList(language="en", chapters=CHAPTERS).model_dump_json())
    (tmp_path / "book.yaml").write_text("title: t\n")
    cfg = BookConfig(title="t", language="en", cast=cast_cfg)
    return paths, cfg


def test_cast_merges_incrementally_and_ranks(tmp_path, monkeypatch):
    paths, cfg = _book(tmp_path, main_cap=2, merge_chapters=1)
    llm = ScriptedLLM(DISCOVER)
    monkeypatch.setattr(cast_stage.Ollama, "from_config", classmethod(lambda cls, cfg, log=None: llm))
    cast_stage.run(paths, cfg)
    cast = paths.load_cast()

    # Aliases merged across batches; distinct people kept.
    assert set(cast.characters) == {"Mr. Darcy", "Elizabeth", "Jane", "the butler"}
    assert cast.characters["Mr. Darcy"].aliases == ["Darcy"]
    assert cast.characters["Elizabeth"].aliases == ["Lizzy"]
    assert cast.resolve("darcy") == "Mr. Darcy"
    # Chapters where each speaks (from discovery, then confirmed by rules).
    assert cast.characters["Mr. Darcy"].chapters == [0, 2]
    assert cast.characters["Elizabeth"].chapters == [0, 1]
    assert cast.characters["Jane"].chapters == [1]
    # Ranked by rule-attributed lines; top main_cap are main; file is in rank order.
    ranked = list(cast.characters)
    assert ranked[0] == "Mr. Darcy" and cast.characters["Mr. Darcy"].lines == 4
    assert [cast.characters[n].main for n in ranked] == [True, True, False, False]
    assert cast.names_for_chapter(1) == ["Mr. Darcy", "Elizabeth", "Jane"]
    assert cast.names_for_chapter(0) == ["Mr. Darcy", "Elizabeth"]
    assert "the butler" in cast.names_for_chapter(2)

    # One discover call per chapter, one merge per chapter (merge_chapters=1),
    # except the batch that resolved by exact name without the model.
    n_discover = sum("SPEAKS" in c for c in llm.calls)
    assert n_discover == 3
    assert (paths.cast_work / "c001.json").exists()

    # Not overwritten without --force.
    paths.cast.write_text("characters: {}\n")
    cast_stage.run(paths, cfg)
    assert paths.cast.read_text() == "characters: {}\n"

    # --force reuses the cached discovery for unchanged chapters.
    before = len(llm.calls)
    cast_stage.run(paths, cfg, force=True)
    assert sum("SPEAKS" in c for c in llm.calls[before:]) == 0
    assert paths.load_cast().characters["Mr. Darcy"].aliases == ["Darcy"]


def test_merge_direct_resolution_needs_no_model():
    cast = Cast(characters={"Ann": Character(aliases=["Annie"])})

    class NoLLM:
        def json(self, *a, **k):
            raise AssertionError("model should not be called")

    cast_stage.merge(cast, [(3, [{"name": "annie", "aliases": ["Miss A"], "description": "d"}])], NoLLM(), "en")
    assert cast.characters["Ann"].aliases == ["Annie", "annie", "Miss A"]
    assert cast.characters["Ann"].chapters == [3] and cast.characters["Ann"].description == "d"


def test_merge_keeps_entries_the_model_dropped():
    cast = Cast()

    class Empty:
        def json(self, *a, **k):
            return {"groups": []}

    cast_stage.merge(cast, [(0, [{"name": "Bob", "aliases": [], "description": "x"}])], Empty(), "en")
    assert list(cast.characters) == ["Bob"] and cast.characters["Bob"].chapters == [0]


def test_legacy_cast_yaml_is_all_main(tmp_path):
    (tmp_path / "cast.yaml").write_text("characters:\n  Ann: {aliases: [], description: d}\n")
    cast = BookPaths(tmp_path).load_cast()
    assert cast.characters["Ann"].main and cast.names_for_chapter(7) == ["Ann"]
