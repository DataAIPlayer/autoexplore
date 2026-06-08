# tests/test_benchmark.py
import json
from pathlib import Path

import pytest

from scripts import benchmark as bm

# 玩具 adapter:纯 Python sleep,无 torch/GPU 依赖;返回 phase1 predictions schema 一条。
TOY_ADAPTER = '''\
import time
def load_model(config):
    return {"delay": config.get("delay", 0.001)}
def infer_one(handle, record):
    time.sleep(handle["delay"])
    return {"image_id": record["image_id"], "score": 0.42}
'''


@pytest.fixture
def bench_run(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    samples = [{"image_id": f"s{i}"} for i in range(5)]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    adapter = tmp_path / "adapter.py"
    adapter.write_text(TOY_ADAPTER)
    return tmp_path, ds, adapter


def test_load_adapter_requires_interface(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def load_model(c): return c\n")     # 缺 infer_one
    with pytest.raises(ValueError):
        bm.load_adapter(bad)


def test_select_records_explicit_subset(bench_run):
    _, ds, _ = bench_run
    cfg = {"subset_ids": ["s3", "s1"]}
    recs = bm.select_records(cfg, ds)
    assert [r["image_id"] for r in recs] == ["s3", "s1"]


def test_select_records_default_first_n_sorted(bench_run):
    _, ds, _ = bench_run
    recs = bm.select_records({"n_records": 3}, ds)
    assert [r["image_id"] for r in recs] == ["s0", "s1", "s2"]


def test_measure_produces_speed_and_predictions(bench_run):
    _, ds, adapter = bench_run
    mod = bm.load_adapter(adapter)
    cfg = {"model_config": {"delay": 0.002}, "warmup": 2, "iters": 3,
           "n_records": 5, "framework": "toy", "gpu_name": "cpu", "gpu_count": 1}
    speed, preds = bm.measure(mod, ds, cfg)
    # 延迟统计字段齐全且合理(每条 sleep 2ms → 均值 ≥ ~2ms)
    for k in ("latency_mean_ms", "p50_ms", "p99_ms", "throughput_qps",
              "n_records", "warmup", "iters", "framework", "gpu_name", "gpu_count"):
        assert k in speed
    assert speed["latency_mean_ms"] >= 1.5          # 宽松下界,避免抖动 flaky
    assert speed["n_records"] == 5
    assert speed["throughput_qps"] > 0
    # predictions 来自首个计时轮,长度 == 子集大小,含 phase1 schema 字段
    assert len(preds) == 5
    assert preds[0]["image_id"] == "s0"
    assert "score" in preds[0]


def test_run_writes_both_files(bench_run):
    base, ds, adapter = bench_run
    cfg_path = base / "bench_config.json"
    cfg_path.write_text(json.dumps({"model_config": {"delay": 0.001},
                                    "warmup": 1, "iters": 2, "n_records": 3}))
    speed = bm.run(adapter, cfg_path, ds,
                   base / "speed.json", base / "predictions.jsonl")
    assert (base / "speed.json").exists()
    pred_lines = (base / "predictions.jsonl").read_text().splitlines()
    assert len(pred_lines) == 3
    assert json.loads(pred_lines[0])["image_id"] == "s0"
    assert json.loads((base / "speed.json").read_text())["n_records"] == 3
