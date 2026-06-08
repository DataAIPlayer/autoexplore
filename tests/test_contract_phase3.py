# tests/test_contract_phase3.py
"""端到端 dry-run:玩具 dataset + 玩具 evaluate.py + 玩具 adapter,
模拟 3a→3b→3c 全链文件流转(baseline → 选 base → 一轮单卡过双门晋升 → 选 final),
断言每步产物 schema 与字段对得上。锁住跨脚本接口。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import phase3_state as p3
from scripts import benchmark as bm
from tests.conftest import TOY_EVALUATE

REPO = Path(__file__).resolve().parent.parent

# 玩具 adapter:延迟由 model_config.delay 决定,质量分由 model_config.score 决定。
TOY_ADAPTER = '''\
import time
def load_model(config):
    return {"delay": config.get("delay", 0.001), "score": config.get("score", 0.40)}
def infer_one(handle, record):
    time.sleep(handle["delay"])
    return {"image_id": record["image_id"], "score": handle["score"]}
'''


@pytest.fixture
def p3_run(tmp_path):
    run = tmp_path / "runs" / "toy-tag"
    ds = run / "dataset"
    ds.mkdir(parents=True)
    samples = [{"image_id": f"s{i}"} for i in range(4)]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    (run / "evaluate.py").write_text(TOY_EVALUATE)
    (run / "phase2").mkdir()
    (run / "phase2" / "state.json").write_text(json.dumps({
        "tag": "toy-tag",
        "backbone": {"id": "p2:winner", "source_dir": "x",
                     "primary_metric": 0.40, "metrics": {}, "version_n": 2}}))
    return run, ds


def _bench_and_score(run, ds, exp_dir, delay, score, bench_cfg):
    """跑玩具 adapter 测速 → speed.json + predictions → 玩具 evaluate → metrics.json。
    返回 (latency_ms, quality)。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    adapter = exp_dir / "adapter.py"
    adapter.write_text(TOY_ADAPTER)
    cfg = dict(bench_cfg, model_config={"delay": delay, "score": score})
    cfg_path = exp_dir / "bench_config.json"
    cfg_path.write_text(json.dumps(cfg))
    speed = bm.run(adapter, cfg_path, ds, exp_dir / "speed.json",
                   exp_dir / "predictions.jsonl")
    metrics_json = exp_dir / "metrics.json"
    subprocess.run([sys.executable, str(run / "evaluate.py"),
                    "--predictions", str(exp_dir / "predictions.jsonl"),
                    "--dataset", str(ds), "--out", str(metrics_json)], check=True)
    quality = json.loads(metrics_json.read_text())["primary_metric"]
    return speed["latency_mean_ms"], quality


def test_full_3a_3b_3c_flow(p3_run):
    run, ds = p3_run
    # iters=5 with n_records=4 → 20 timed samples per measurement, stable means.
    # Delays are spaced ≥2× apart so mean ordering holds despite OS scheduling noise.
    bench_cfg = {"warmup": 2, "iters": 5, "n_records": 4, "framework": "toy"}

    # 0. init + baseline(主干原生:慢≈10ms,质量 0.40)
    state = p3.init_state(run, "toy-tag")
    base_lat, base_q = _bench_and_score(run, ds, run / "phase3" / "baseline",
                                        0.010, 0.40, bench_cfg)
    state = p3.set_baseline(run, state, base_lat, 1000.0 / base_lat, quality=base_q)
    assert p3.next_action(state)["action"] == "framework_select"

    # 3a. 两个框架:
    #   vllm(≈4ms, 质量达标 0.399) — 质量 passes_quality, 速度 >2× base
    #   sglang(≈2ms, 质量崩 0.30) — 最快但质量不达标, 应被排除
    for name, delay, score in [("vllm", 0.004, 0.399), ("sglang", 0.002, 0.30)]:
        p3.framework_add(run, state, name)
        lat, q = _bench_and_score(run, ds, run / "phase3" / "frameworks" / name,
                                  delay, score, bench_cfg)
        p3.framework_record(run, state, name, lat, q, "ready")
    state = p3.select_base(run, state)
    assert state["base_framework"]["name"] == "vllm"        # sglang 质量不达标被排除
    assert state["sub_phase"] == "single-card-loop"

    # 3b. 一轮单卡:方案 a 在 vllm base(≈4ms)上再快一截(≈2ms → >10% 提速)且质量达标
    # 2ms vs 4ms → ~50% speedup, well above the 10% gate threshold
    state = p3.open_round(run, state, [{"slot": "a", "tier": "quantization"}])
    lat, q = _bench_and_score(run, ds,
                              run / "phase3" / "single_card" / "rounds" / "r001" / "exp_a",
                              0.002, 0.398, bench_cfg)
    state = p3.record_slot(run, state, "r001", "a", lat, q, "done")
    slot = state["rounds"][-1]["slots"][0]
    accept = p3.passes_gate(state["base_framework"]["latency_ms"], lat,
                            state["baseline"]["quality"], q)
    assert accept is True
    state = p3.promote_base(run, state, "fp8-quant",
                            "single_card/rounds/r001/exp_a", lat, q)
    p3.append_phase3_result(run, "round", "fp8-quant",
                            state["base_framework"]["version_n"], lat,
                            slot["speedup_pct"], q, slot["quality_loss_pct"],
                            "accepted", "FP8 on vllm")
    assert state["base_framework"]["version_n"] == 1

    # 饱和 → 进 3c
    assert p3.saturation_check(run, state, force=True) is True
    assert p3.next_action(state)["action"] == "parallel"

    # 3c. 两个并行方案:tp2(≈2ms, 更快)、pp2(≈6ms, 较慢);均质量达标
    # tp2 delay is 3× smaller than pp2 → unambiguous ordering across any OS scheduling noise
    for name, delay, gpus in [("tp2", 0.002, 2), ("pp2", 0.006, 2)]:
        p3.parallel_add(run, state, name, gpu_count=gpus)
        lat, q = _bench_and_score(run, ds,
                                  run / "phase3" / "multi_card" / f"scheme_{name}",
                                  delay, 0.399, bench_cfg)
        p3.parallel_record(run, state, name, lat, q, "ready")
    state = p3.select_final(run, state)
    assert state["final"]["scheme"] == "tp2"
    assert state["final"]["gpu_count"] == 2
    assert state["sub_phase"] == "done"
    assert p3.next_action(state)["action"] == "done"

    # results.tsv 有表头 + 至少一行
    lines = (run / "phase3" / "results.tsv").read_text().splitlines()
    assert lines[0].split("\t") == p3.PHASE3_RESULTS_HEADER
    assert len(lines) >= 2
