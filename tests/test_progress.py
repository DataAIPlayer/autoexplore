import json
from scripts.progress import (
    load_progress, save_progress, Progress,
    append_result, read_results, is_done,
)

def test_save_then_load_roundtrip(tmp_path):
    p = Progress(model="llava", stage="C", retry_count=1,
                 gpus="2,5", image_tag="autoexplore/llava",
                 last_error="boom")
    save_progress(tmp_path, p)
    loaded = load_progress(tmp_path)
    assert loaded.model == "llava"
    assert loaded.stage == "C"
    assert loaded.retry_count == 1
    assert loaded.gpus == "2,5"
    assert loaded.last_error == "boom"

def test_save_sets_updated_at(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="A"))
    raw = json.loads((tmp_path / "progress.json").read_text())
    assert raw["updated_at"]  # 非空时间戳

def test_load_missing_returns_none(tmp_path):
    assert load_progress(tmp_path) is None

def test_is_done_true_for_terminal_stages(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="ready"))
    assert is_done(tmp_path) is True
    save_progress(tmp_path, Progress(model="m", stage="crash"))
    assert is_done(tmp_path) is True

def test_is_done_false_for_in_progress(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="B"))
    assert is_done(tmp_path) is False

def test_is_done_false_when_no_progress(tmp_path):
    assert is_done(tmp_path) is False

def test_append_result_writes_header_once(tmp_path):
    append_result(tmp_path, "llava", 0.91, 44.0, "ready", "baseline")
    append_result(tmp_path, "blip", 0.88, 40.0, "ready", "second model")
    rows = read_results(tmp_path)
    assert rows[0] == ["model", "primary_metric", "memory_gb", "status", "description"]
    assert rows[1] == ["llava", "0.910000", "44.0", "ready", "baseline"]
    assert rows[2] == ["blip", "0.880000", "40.0", "ready", "second model"]

def test_append_result_crash_formatting(tmp_path):
    append_result(tmp_path, "x", 0.0, 0.0, "crash", "OOM")
    rows = read_results(tmp_path)
    assert rows[1] == ["x", "0.000000", "0.0", "crash", "OOM"]
