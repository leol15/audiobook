"""Run with: uv run --project backends/qwen3tts --with pytest python -m pytest backends/qwen3tts (no model needed)."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("worker", pathlib.Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_token_cap_scales_with_longest_text():
    short = worker.token_cap(["你好。"], "Chinese")
    long = worker.token_cap(["你好。", "一" * 134], "Chinese")
    assert short < long < 2048
    assert long == int(134 / 4.0 * 12 * 2.0) + 48
    assert worker.token_cap(["a" * 300], "English") == int(300 / 15.0 * 12 * 2.0) + 48
