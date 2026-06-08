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


def test_framework_record_sets_passes_quality(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    p3.set_baseline(run_with_phase2, state, 100.0, 10.0)        # 基线质量 0.40
    p3.framework_add(run_with_phase2, state, "vllm")
    state = p3.framework_record(run_with_phase2, state, "vllm",
                                latency_ms=60.0, quality=0.398, status="ready")
    fw = state["frameworks"][0]
    assert fw["passes_quality"] is True            # 0.5% 损失,达标
    assert fw["latency_ms"] == pytest.approx(60.0)
    assert fw["status"] == "ready"


def test_select_base_picks_fastest_quality_passing(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    p3.set_baseline(run_with_phase2, state, 100.0, 10.0)
    for name, lat, q in [("vllm", 55.0, 0.30), ("sglang", 70.0, 0.398),
                         ("trtllm", 50.0, 0.399)]:
        p3.framework_add(run_with_phase2, state, name)
        p3.framework_record(run_with_phase2, state, name, lat, q, "ready")
    state = p3.select_base(run_with_phase2, state)
    assert state["base_framework"]["name"] == "trtllm"   # 50ms 且质量达标
    assert state["base_framework"]["version_n"] == 0
    assert state["sub_phase"] == "single-card-loop"


def test_select_base_falls_back_to_native_when_none_pass(run_with_phase2):
    state = p3.init_state(run_with_phase2, "toy-tag")
    p3.set_baseline(run_with_phase2, state, 100.0, 10.0)
    p3.framework_add(run_with_phase2, state, "vllm")
    p3.framework_record(run_with_phase2, state, "vllm", 50.0, 0.30, "ready")  # 快但质量崩
    state = p3.select_base(run_with_phase2, state)
    assert state["base_framework"]["name"] == "native"
    assert state["base_framework"]["latency_ms"] == pytest.approx(100.0)  # 退回主干原生延迟
    assert state["sub_phase"] == "single-card-loop"


def _ready_base(run_dir):
    """helper:init + baseline + 选出 base(60ms/质量0.398)。"""
    state = p3.init_state(run_dir, "toy-tag")
    p3.set_baseline(run_dir, state, 100.0, 10.0)
    p3.framework_add(run_dir, state, "vllm")
    p3.framework_record(run_dir, state, "vllm", 60.0, 0.398, "ready")
    return p3.select_base(run_dir, state)


def test_open_round_creates_slots(run_with_phase2):
    state = _ready_base(run_with_phase2)
    dirs = [{"slot": "a", "tier": "quantization"},
            {"slot": "b", "tier": "compile"},
            {"slot": "c", "tier": "decoding"}]
    state = p3.open_round(run_with_phase2, state, dirs)
    rnd = state["rounds"][-1]
    assert rnd["id"] == "r001"
    assert [s["slot"] for s in rnd["slots"]] == ["a", "b", "c"]
    assert rnd["slots"][0]["exp_dir"] == "single_card/rounds/r001/exp_a"


def test_record_slot_computes_speedup_and_loss(run_with_phase2):
    state = _ready_base(run_with_phase2)                 # base 延迟 60ms,基线质量 0.40
    state = p3.open_round(run_with_phase2, state,
                          [{"slot": "a", "tier": "quantization"}])
    state = p3.record_slot(run_with_phase2, state, "r001", "a",
                           latency_ms=48.0, quality=0.398, status="done")
    slot = state["rounds"][-1]["slots"][0]
    assert slot["speedup_pct"] == pytest.approx(20.0)   # (60-48)/60
    assert slot["quality_loss_pct"] == pytest.approx(0.5, abs=1e-6)  # (0.40-0.398)/0.40
    assert state["rounds"][-1]["status"] == "scored"    # 全 slot 终态


def test_promote_base_bumps_version_and_resets_dry_streak(run_with_phase2):
    state = _ready_base(run_with_phase2)
    state["dry_streak"] = 2
    state = p3.promote_base(run_with_phase2, state, "fp8-quant",
                            "single_card/rounds/r001/exp_a", 48.0, 0.398)
    assert state["base_framework"]["version_n"] == 1
    assert state["base_framework"]["name"] == "fp8-quant"
    assert state["base_framework"]["latency_ms"] == pytest.approx(48.0)
    assert state["dry_streak"] == 0                      # 晋升清零饱和计数
    assert len(state["base_history"]) == 2


def test_bump_dry_streak(run_with_phase2):
    state = _ready_base(run_with_phase2)
    state = p3.bump_dry_streak(run_with_phase2, state)
    state = p3.bump_dry_streak(run_with_phase2, state)
    assert state["dry_streak"] == 2


def test_saturation_advances_to_multi_card_at_k(run_with_phase2):
    state = _ready_base(run_with_phase2)              # saturation_k 默认 3
    for _ in range(2):
        p3.bump_dry_streak(run_with_phase2, state)
    advanced = p3.saturation_check(run_with_phase2, state)
    assert advanced is False                          # 2 < 3,不进
    assert state["sub_phase"] == "single-card-loop"
    p3.bump_dry_streak(run_with_phase2, state)        # → 3
    advanced = p3.saturation_check(run_with_phase2, state)
    assert advanced is True
    assert state["sub_phase"] == "multi-card"


def test_saturation_manual_trigger(run_with_phase2):
    state = _ready_base(run_with_phase2)
    advanced = p3.saturation_check(run_with_phase2, state, force=True)
    assert advanced is True                           # 人工提前触发
    assert state["sub_phase"] == "multi-card"
