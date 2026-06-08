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
