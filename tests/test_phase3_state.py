# tests/test_phase3_state.py
import json
from pathlib import Path

import pytest

from scripts import phase3_state as p3


def test_speedup_ratio():
    assert p3.speedup_ratio(100.0, 90.0) == pytest.approx(0.10)
    assert p3.speedup_ratio(100.0, 100.0) == pytest.approx(0.0)
    assert p3.speedup_ratio(0.0, 50.0) == 0.0          # base 延迟为 0 → 0


def test_quality_loss_ratio():
    assert p3.quality_loss_ratio(0.40, 0.396) == pytest.approx(0.01)
    assert p3.quality_loss_ratio(0.40, 0.42) == pytest.approx(-0.05)   # 更好 → 负损失
    assert p3.quality_loss_ratio(0.0, 0.1) == 0.0      # 基线为 0 → 0


def test_passes_quality_boundary():
    assert p3.passes_quality(0.40, 0.396) is True      # 恰好 1% 损失,达标
    assert p3.passes_quality(0.40, 0.395) is False     # 1.25% 损失,超
    assert p3.passes_quality(0.40, 0.50) is True        # 更好,达标


def test_passes_gate_dual_AND():
    # base_lat=100, baseline_q=0.40
    assert p3.passes_gate(100.0, 90.0, 0.40, 0.396) is True    # 提速10% 且 损失1% → 过
    assert p3.passes_gate(100.0, 91.0, 0.40, 0.40) is False    # 提速9% → 不过
    assert p3.passes_gate(100.0, 90.0, 0.40, 0.39) is False    # 损失2.5% → 不过
    assert p3.passes_gate(100.0, 89.0, 0.40, 0.42) is True     # 提速11% 且质量更好 → 过


def test_best_by_latency_picks_min_among_quality_passing():
    cands = [
        {"name": "f1", "latency_ms": 80.0, "passes_quality": True},
        {"name": "f2", "latency_ms": 50.0, "passes_quality": False},   # 最快但质量不达标
        {"name": "f3", "latency_ms": 70.0, "passes_quality": True},
    ]
    best = p3.best_by_latency(cands)
    assert best["name"] == "f3"


def test_best_by_latency_none_when_no_quality_passing():
    cands = [{"name": "f1", "latency_ms": 50.0, "passes_quality": False}]
    assert p3.best_by_latency(cands) is None


@pytest.fixture
def run_with_phase2(tmp_path):
    """造一个 run_dir,含 phase2/state.json(backbone=p2:winner, primary_metric=0.40)。"""
    run = tmp_path / "runs" / "toy-tag"
    (run / "phase2").mkdir(parents=True)
    (run / "phase2" / "state.json").write_text(json.dumps({
        "tag": "toy-tag",
        "backbone": {"id": "p2:winner", "source_dir": "phase2/rounds/r003/exp_a",
                     "primary_metric": 0.40, "metrics": {"score_mean": 0.40}, "version_n": 3},
    }))
    return run


def test_init_reads_phase2_backbone(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    assert state["sub_phase"] == "framework-select"
    assert state["baseline"]["source"] == "phase2-backbone:p2:winner"
    assert state["baseline"]["quality"] == pytest.approx(0.40)
    assert state["baseline"]["latency_ms"] is None      # 尚未测速
    assert state["dry_streak"] == 0
    assert state["saturation_k"] == p3.SATURATION_K_DEFAULT
    # 落盘可重载
    assert p3.load_state(run_with_phase2)["sub_phase"] == "framework-select"


def test_set_baseline_fills_latency(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    state = p3.set_baseline(run_with_phase2, state, latency_ms=100.0, throughput_qps=10.0)
    assert state["baseline"]["latency_ms"] == pytest.approx(100.0)
    assert state["baseline"]["throughput_qps"] == pytest.approx(10.0)
    assert state["baseline"]["quality"] == pytest.approx(0.40)   # 沿用 phase2 分


def test_save_state_atomic_roundtrip(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    state["frameworks"].append({"name": "vllm"})
    p3.save_state(run_with_phase2, state)
    assert p3.load_state(run_with_phase2)["frameworks"][0]["name"] == "vllm"
