# autoexplore 第三阶段(推理速度优化)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 autoexplore 第三阶段「模型推理速度优化」skill —— 取 phase2 获胜主干,经 3a 单卡框架选型 → 3b 单卡加速方案双门优化循环 → 3c 多卡并行扩展,产出最终生产级多卡推理方案。

**Architecture:** 混合形态(对齐 phase1/2):新 skill `autoexplore-phase3` 负责判断与编排;两个新 Python 脚本承担确定性重活 —— `phase3_state.py`(状态机:base 指针 / 双门数学 / 饱和判定 / 中断恢复)与 `benchmark.py`(冻结测速协议:warmup + batch=1 单条计时,一次执行同产 speed.json + predictions.jsonl)。唯一可变部分是 Claude 按框架/方案写的 `adapter.py`。复用 phase1/2 的 `gpu_select / docker_env / run_inference / compute_metrics / progress / directions_schema / train_launch`。

**Tech Stack:** Python 3(标准库 + `from __future__ import annotations`)、pytest、uv、Docker + NVIDIA runtime(运行时)。脚本零三方依赖;测速的 CUDA sync 对 torch 做可选保护(单测无 GPU 可跑)。

**设计依据:** [docs/superpowers/specs/2026-06-08-autoexplore-phase3-speedup-design.md](../specs/2026-06-08-autoexplore-phase3-speedup-design.md)

---

## 文件结构

**新建:**
- `scripts/phase3_state.py` —— 状态机:双门纯函数、baseline、framework 登记与选型(3a)、base 晋升与轮次(3b)、dry_streak 饱和(3b→3c)、parallel 选型(3c)、resume、results.tsv、CLI。
- `scripts/benchmark.py` —— 冻结测速协议:动态加载 adapter、固定子集、warmup+计时、speed.json + predictions.jsonl 双产出、CLI。
- `tests/test_phase3_state.py` —— phase3_state 单元测试。
- `tests/test_benchmark.py` —— benchmark 单元测试(玩具 sleep adapter,无 GPU)。
- `tests/test_contract_phase3.py` —— 端到端 dry-run 契约测试(3a→3b→3c 文件流转)。
- `tests/test_smoke_phase3_gpu.py` —— `@pytest.mark.gpu` 冒烟,默认 skip。
- `skills/autoexplore-phase3/SKILL.md` —— skill 入口。
- `skills/autoexplore-phase3/references/speedup-loop.md` —— 三模式细则。

**修改:**
- `scripts/directions_schema.py` —— 扩展 tier 集合支持 phase3,新增 `--tiers` 选择(默认 phase2,向后兼容)。
- `tests/test_directions_schema.py` —— 增 phase3 tier 用例。

**复用(不改):** `gpu_select.py` / `docker_env.py` / `run_inference.py` / `compute_metrics.py` / `progress.py` / `train_launch.py`。

**约定(全程):** 所有命令从 repo 根 `uv run` 执行;state.json 经 `phase3_state.py` 独占读写;phase3 产物落 `runs/<tag>/phase3/`。

---

## Task 1: phase3_state — 双门纯函数

**Files:**
- Create: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'scripts.phase3_state'`

- [ ] **Step 3: 写最小实现**

```python
# scripts/phase3_state.py
"""Phase-3 状态机 (phase3/state.json 唯一真相源):base 指针 / 双门数学 /
框架与方案登记 / 饱和判定 / 空闲卡并发派发 / 中断恢复。本模块独占读写 state.json。"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# 复用 phase2 的通用派发与方向指纹(DRY:phase3 本就依赖 phase2 产物)
from scripts.phase2_state import _direction_fingerprint, plan_dispatch

MIN_SPEEDUP = 0.10          # 相对当前 base 提速 ≥10%
MAX_QUALITY_LOSS = 0.01     # 相对最初基线 质量损失 ≤1%
SATURATION_K_DEFAULT = 3    # 连续 K 轮无方案过门 → 进多卡
_EPS = 1e-9


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def speedup_ratio(base_lat: float, cand_lat: float) -> float:
    """相对提速 = (base - cand) / base。base≤0 时返回 0(无意义基准)。"""
    if base_lat <= 0.0:
        return 0.0
    return (base_lat - cand_lat) / base_lat


def quality_loss_ratio(baseline_q: float, cand_q: float) -> float:
    """相对质量损失 = (baseline - cand) / baseline。可为负(候选更好)。基线≤0 返回 0。"""
    if baseline_q <= 0.0:
        return 0.0
    return (baseline_q - cand_q) / baseline_q


def passes_quality(baseline_q: float, cand_q: float,
                   max_loss: float = MAX_QUALITY_LOSS) -> bool:
    return quality_loss_ratio(baseline_q, cand_q) <= max_loss + _EPS


def passes_gate(base_lat: float, cand_lat: float, baseline_q: float, cand_q: float,
                min_speedup: float = MIN_SPEEDUP,
                max_loss: float = MAX_QUALITY_LOSS) -> bool:
    """双门 AND:相对当前 base 提速 ≥min_speedup 且 相对基线质量损失 ≤max_loss。"""
    fast_enough = speedup_ratio(base_lat, cand_lat) >= min_speedup - _EPS
    return fast_enough and passes_quality(baseline_q, cand_q, max_loss)


def best_by_latency(candidates: list[dict]) -> dict | None:
    """3a/3c 选型:质量达标(passes_quality=True)候选里 latency_ms 最小者;无则 None。"""
    eligible = [c for c in candidates if c.get("passes_quality")]
    if not eligible:
        return None
    return min(eligible, key=lambda c: c["latency_ms"])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): add dual-gate pure functions to phase3_state"
```

---

## Task 2: phase3_state — state 读写 + 从 phase2 init + baseline

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_phase3_state.py`)

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_init_reads_phase2_backbone -v`
Expected: FAIL —— `AttributeError: module 'scripts.phase3_state' has no attribute 'init_state'`

- [ ] **Step 3: 写最小实现**(追加到 `scripts/phase3_state.py`)

```python
def load_state(run_dir: Path) -> dict | None:
    p = run_dir / "phase3" / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_state(run_dir: Path, state: dict) -> None:
    p = run_dir / "phase3" / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(p)  # 原子替换


def backbone_from_phase2(run_dir: Path) -> tuple[str, float]:
    """读 phase2/state.json 的 backbone,返回 (id, primary_metric)。无则抛 FileNotFoundError/KeyError。"""
    st = json.loads((run_dir / "phase2" / "state.json").read_text())
    bb = st["backbone"]
    return bb["id"], float(bb["primary_metric"])


def init_state(run_dir: Path, tag: str) -> dict:
    bb_id, bb_metric = backbone_from_phase2(run_dir)
    state = {
        "tag": tag,
        "baseline": {"source": f"phase2-backbone:{bb_id}", "quality": bb_metric,
                     "latency_ms": None, "throughput_qps": None},
        "sub_phase": "framework-select",
        "frameworks": [],
        "base_framework": None,
        "base_history": [],
        "round_counter": 0,
        "rounds": [],
        "dry_streak": 0,
        "saturation_k": SATURATION_K_DEFAULT,
        "directions_tried": [],
        "parallel_schemes": [],
        "final": None,
        "updated_at": _now(),
    }
    save_state(run_dir, state)
    return state


def set_baseline(run_dir: Path, state: dict, latency_ms: float,
                 throughput_qps: float, quality: float | None = None) -> dict:
    state["baseline"]["latency_ms"] = latency_ms
    state["baseline"]["throughput_qps"] = throughput_qps
    if quality is not None:                       # 可选:用实测质量纠正 phase2 占位
        state["baseline"]["quality"] = quality
    save_state(run_dir, state)
    return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(9 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): init state from phase2 backbone + baseline setter"
```

---

## Task 3: phase3_state — 框架登记与 3a 选型

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_select_base_picks_fastest_quality_passing -v`
Expected: FAIL —— `AttributeError: ... 'framework_add'`

- [ ] **Step 3: 写最小实现**

```python
def _find_framework(state: dict, name: str) -> dict:
    for f in state["frameworks"]:
        if f["name"] == name:
            return f
    raise KeyError(name)


def framework_add(run_dir: Path, state: dict, name: str) -> dict:
    state["frameworks"].append({"name": name, "exp_dir": f"frameworks/{name}",
                                "status": "pending", "latency_ms": None,
                                "quality": None, "passes_quality": False})
    save_state(run_dir, state)
    return state


def framework_record(run_dir: Path, state: dict, name: str, latency_ms: float,
                     quality: float, status: str) -> dict:
    fw = _find_framework(state, name)
    fw["latency_ms"] = latency_ms
    fw["quality"] = quality
    fw["status"] = status                          # ready|crash
    fw["passes_quality"] = passes_quality(state["baseline"]["quality"], quality)
    save_state(run_dir, state)
    return state


def select_base(run_dir: Path, state: dict) -> dict:
    """3a:质量达标框架里最小延迟者 = base v0;无则退回主干原生推理。推进 sub_phase。"""
    ready = [f for f in state["frameworks"] if f["status"] == "ready"]
    winner = best_by_latency(ready)
    if winner is None:
        base = {"name": "native", "exp_dir": "baseline",
                "latency_ms": state["baseline"]["latency_ms"],
                "quality": state["baseline"]["quality"], "version_n": 0}
    else:
        base = {"name": winner["name"], "exp_dir": winner["exp_dir"],
                "latency_ms": winner["latency_ms"], "quality": winner["quality"],
                "version_n": 0}
    state["base_framework"] = base
    state["base_history"] = [{"version_n": 0, "name": base["name"],
                              "latency_ms": base["latency_ms"], "promoted_at": _now()}]
    state["sub_phase"] = "single-card-loop"
    save_state(run_dir, state)
    return state


def base_get(state: dict) -> dict | None:
    return state.get("base_framework")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(12 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): framework registry + 3a base selection"
```

---

## Task 4: phase3_state — 3b 轮次、slot 记账、base 晋升

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_record_slot_computes_speedup_and_loss -v`
Expected: FAIL —— `AttributeError: ... 'open_round'`

- [ ] **Step 3: 写最小实现**

```python
def _find_round(state: dict, round_id: str) -> dict:
    for r in state["rounds"]:
        if r["id"] == round_id:
            return r
    raise KeyError(round_id)


def open_round(run_dir: Path, state: dict, directions: list[dict]) -> dict:
    state["round_counter"] += 1
    rid = f"r{state['round_counter']:03d}"
    slots = [{"slot": d["slot"],
              "exp_dir": f"single_card/rounds/{rid}/exp_{d['slot']}",
              "tier": d.get("tier", ""), "latency_ms": None, "quality": None,
              "speedup_pct": None, "quality_loss_pct": None, "status": "pending"}
             for d in directions]
    state["rounds"].append({"id": rid, "status": "open", "slots": slots})
    save_state(run_dir, state)
    return state


def record_slot(run_dir: Path, state: dict, round_id: str, slot: str,
                latency_ms: float, quality: float, status: str) -> dict:
    """status ∈ {done, crash}。done 的 slot 算 speedup% vs 当前 base、loss% vs 基线。
    全 slot 终态则轮转 scored。slot 不存在抛 KeyError(fail loud)。"""
    rnd = _find_round(state, round_id)
    target = next((s for s in rnd["slots"] if s["slot"] == slot), None)
    if target is None:
        raise KeyError(f"slot {slot!r} not in round {round_id!r}")
    base_lat = state["base_framework"]["latency_ms"]
    baseline_q = state["baseline"]["quality"]
    target["latency_ms"] = latency_ms
    target["quality"] = quality
    target["speedup_pct"] = speedup_ratio(base_lat, latency_ms) * 100.0
    target["quality_loss_pct"] = quality_loss_ratio(baseline_q, quality) * 100.0
    target["status"] = status
    rnd["status"] = ("scored"
                     if all(s["status"] in ("done", "crash") for s in rnd["slots"])
                     else "running")
    save_state(run_dir, state)
    return state


def promote_base(run_dir: Path, state: dict, name: str, exp_dir: str,
                 latency_ms: float, quality: float) -> dict:
    v = state["base_framework"]["version_n"] + 1
    state["base_framework"] = {"name": name, "exp_dir": exp_dir,
                               "latency_ms": latency_ms, "quality": quality,
                               "version_n": v}
    state["base_history"].append({"version_n": v, "name": name,
                                  "latency_ms": latency_ms, "promoted_at": _now()})
    state["dry_streak"] = 0
    save_state(run_dir, state)
    return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(16 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): 3b rounds, slot scoring, base promotion"
```

---

## Task 5: phase3_state — dry_streak 饱和与 3b→3c 过渡

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_saturation_advances_to_multi_card_at_k -v`
Expected: FAIL —— `AttributeError: ... 'bump_dry_streak'`

- [ ] **Step 3: 写最小实现**

```python
def bump_dry_streak(run_dir: Path, state: dict) -> dict:
    state["dry_streak"] += 1
    save_state(run_dir, state)
    return state


def saturation_check(run_dir: Path, state: dict, force: bool = False) -> bool:
    """dry_streak ≥ saturation_k(或 force)→ sub_phase=multi-card,返回 True;否则 False。"""
    if force or state["dry_streak"] >= state["saturation_k"]:
        state["sub_phase"] = "multi-card"
        save_state(run_dir, state)
        return True
    return False
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(19 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): dry-streak saturation + 3b->3c transition"
```

---

## Task 6: phase3_state — 3c 多卡方案登记与终选

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
def test_parallel_record_and_select_final(run_with_phase2):
    state = _ready_base(run_with_phase2)
    p3.saturation_check(run_with_phase2, state, force=True)     # 进 multi-card
    for name, lat, q, gpus in [("tp2", 35.0, 0.399, 2),
                               ("pp2", 45.0, 0.40, 2),
                               ("tp4", 30.0, 0.30, 4)]:          # tp4 最快但质量崩
        p3.parallel_add(run_with_phase2, state, name, gpu_count=gpus)
        p3.parallel_record(run_with_phase2, state, name, lat, q, "ready")
    state = p3.select_final(run_with_phase2, state)
    assert state["final"]["scheme"] == "tp2"                    # 35ms 且质量达标
    assert state["final"]["gpu_count"] == 2
    assert state["sub_phase"] == "done"


def test_select_final_none_keeps_single_card(run_with_phase2):
    state = _ready_base(run_with_phase2)
    p3.saturation_check(run_with_phase2, state, force=True)
    p3.parallel_add(run_with_phase2, state, "tp4", gpu_count=4)
    p3.parallel_record(run_with_phase2, state, "tp4", 30.0, 0.20, "ready")  # 质量崩
    state = p3.select_final(run_with_phase2, state)
    # 无并行方案达标 → final 回落单卡 base,标 gpu_count=1
    assert state["final"]["scheme"] == state["base_framework"]["name"]
    assert state["final"]["gpu_count"] == 1
    assert state["sub_phase"] == "done"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_parallel_record_and_select_final -v`
Expected: FAIL —— `AttributeError: ... 'parallel_add'`

- [ ] **Step 3: 写最小实现**

```python
def _find_parallel(state: dict, name: str) -> dict:
    for s in state["parallel_schemes"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


def parallel_add(run_dir: Path, state: dict, name: str, gpu_count: int) -> dict:
    state["parallel_schemes"].append({"name": name, "exp_dir": f"multi_card/scheme_{name}",
                                      "gpu_count": gpu_count, "status": "pending",
                                      "latency_ms": None, "quality": None,
                                      "passes_quality": False})
    save_state(run_dir, state)
    return state


def parallel_record(run_dir: Path, state: dict, name: str, latency_ms: float,
                    quality: float, status: str) -> dict:
    sch = _find_parallel(state, name)
    sch["latency_ms"] = latency_ms
    sch["quality"] = quality
    sch["status"] = status
    sch["passes_quality"] = passes_quality(state["baseline"]["quality"], quality)
    save_state(run_dir, state)
    return state


def select_final(run_dir: Path, state: dict) -> dict:
    """3c:质量达标并行方案里最小延迟者 = final;无则回落单卡 base(gpu_count=1)。sub_phase=done。"""
    ready = [s for s in state["parallel_schemes"] if s["status"] == "ready"]
    winner = best_by_latency(ready)
    if winner is None:
        base = state["base_framework"]
        state["final"] = {"scheme": base["name"], "exp_dir": base["exp_dir"],
                          "latency_ms": base["latency_ms"], "quality": base["quality"],
                          "gpu_count": 1}
    else:
        state["final"] = {"scheme": winner["name"], "exp_dir": winner["exp_dir"],
                          "latency_ms": winner["latency_ms"], "quality": winner["quality"],
                          "gpu_count": winner["gpu_count"]}
    state["sub_phase"] = "done"
    save_state(run_dir, state)
    return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(21 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): 3c parallel schemes + final selection"
```

---

## Task 7: phase3_state — 方向去重、派发、resume

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
def test_directions_dedup(run_with_phase2):
    state = _ready_base(run_with_phase2)
    d = {"title": "FP8 quant", "source_urls": ["http://a", "http://b"]}
    assert p3.directions_seen(state, d) is False
    p3.directions_add(run_with_phase2, state, [d])
    # url 顺序无关、大小写归一仍判已见
    d2 = {"title": "fp8 quant", "source_urls": ["http://b", "http://a"]}
    assert p3.directions_seen(state, d2) is True


def test_plan_dispatch_reused(run_with_phase2):
    exps = [{"slot": "a", "needs_gpus": 1, "is_training": False},
            {"slot": "b", "needs_gpus": 2, "is_training": True}]
    out = p3.plan_dispatch(exps, [0, 1])
    assert out["assigned"]["a"] == [0]          # 非训练优先
    assert out["queued"] == ["b"]               # 训练型卡不够排队


def test_next_action_resume_across_subphases(run_with_phase2):
    assert p3.next_action(None)["action"] == "init"
    state = p3.init_state(run_with_phase2, "toy-tag")
    assert p3.next_action(state)["action"] == "baseline"      # 尚未测基线
    p3.set_baseline(run_with_phase2, state, 100.0, 10.0)
    assert p3.next_action(state)["action"] == "framework_select"
    state = _ready_base(run_with_phase2)                       # 已选 base,进单卡环
    assert p3.next_action(state)["action"] == "search"
    p3.open_round(run_with_phase2, state, [{"slot": "a", "tier": "compile"}])
    assert p3.next_action(state)["action"] == "execute"
    p3.record_slot(run_with_phase2, state, "r001", "a", 48.0, 0.398, "done")
    assert p3.next_action(state)["action"] == "gate_check"
    p3.saturation_check(run_with_phase2, state, force=True)
    assert p3.next_action(state)["action"] == "parallel"
    p3.select_final(run_with_phase2, state)
    assert p3.next_action(state)["action"] == "done"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_next_action_resume_across_subphases -v`
Expected: FAIL —— `AttributeError: ... 'directions_seen'`

- [ ] **Step 3: 写最小实现**

```python
def directions_seen(state: dict, direction: dict) -> bool:
    return _direction_fingerprint(direction) in state.get("directions_tried", [])


def directions_add(run_dir: Path, state: dict, directions: list[dict]) -> dict:
    for d in directions:
        fp = _direction_fingerprint(d)
        if fp not in state["directions_tried"]:
            state["directions_tried"].append(fp)
    save_state(run_dir, state)
    return state


def next_action(state: dict | None) -> dict:
    """中断恢复:读 state 给下一步动作(按 sub_phase)。"""
    if not state:
        return {"action": "init"}
    sp = state.get("sub_phase")
    if sp == "framework-select":
        if state["baseline"].get("latency_ms") in (None, 0):
            return {"action": "baseline"}
        return {"action": "framework_select"}
    if sp == "single-card-loop":
        rounds = state.get("rounds", [])
        if not rounds or rounds[-1]["status"] == "done":
            return {"action": "search"}
        last = rounds[-1]
        if last["status"] == "scored":
            return {"action": "gate_check", "round": last["id"]}
        return {"action": "execute", "round": last["id"]}
    if sp == "multi-card":
        return {"action": "parallel"}
    if sp == "done":
        return {"action": "done"}
    return {"action": "init"}
```

注:`plan_dispatch` 已在 Task 1 顶部从 `phase2_state` 导入,无需重写。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(24 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): directions dedup, dispatch reuse, resume next_action"
```

---

## Task 8: phase3_state — results.tsv + CLI

**Files:**
- Modify: `scripts/phase3_state.py`
- Test: `tests/test_phase3_state.py`

- [ ] **Step 1: 写失败测试**

```python
import subprocess
import sys


def test_append_result_writes_header_then_row(run_with_phase2):
    p3.append_phase3_result(run_with_phase2, stage="round", name="fp8-quant",
                            base_ver=1, latency_ms=48.0, speedup_pct=20.0,
                            quality=0.398, quality_loss_pct=0.5, status="accepted",
                            description="FP8 weight-only")
    lines = (run_with_phase2 / "phase3" / "results.tsv").read_text().splitlines()
    assert lines[0].split("\t")[0] == "stage"
    assert lines[1].split("\t")[1] == "fp8-quant"


def test_cli_gate_dual(run_with_phase2):
    out = subprocess.run(
        [sys.executable, "-m", "scripts.phase3_state", "gate",
         "--base-lat", "100", "--cand-lat", "90",
         "--baseline-q", "0.40", "--cand-q", "0.398"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert json.loads(out.stdout)["accept"] is True


def test_cli_resume(run_with_phase2):
    p3.init_state(run_with_phase2, "toy-tag")
    out = subprocess.run(
        [sys.executable, "-m", "scripts.phase3_state", "resume",
         "--run-dir", str(run_with_phase2)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert json.loads(out.stdout)["action"] == "baseline"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_phase3_state.py::test_cli_gate_dual -v`
Expected: FAIL —— `AttributeError: ... 'append_phase3_result'`(或 CLI 无 gate 子命令)

- [ ] **Step 3: 写最小实现**

```python
PHASE3_RESULTS_HEADER = ["stage", "name", "base_ver", "latency_ms", "speedup_pct",
                         "quality", "quality_loss_pct", "status", "description"]


def append_phase3_result(run_dir: Path, stage: str, name: str, base_ver: int,
                         latency_ms: float, speedup_pct: float | None, quality: float,
                         quality_loss_pct: float | None, status: str,
                         description: str) -> None:
    """追加一行 phase3/results.tsv(append-only 研究日志,untracked)。
    stage ∈ {framework, round, parallel};status ∈ {accepted, discard, crash, ready}。"""
    path = run_dir / "phase3" / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if is_new:
            w.writerow(PHASE3_RESULTS_HEADER)
        sp = "" if speedup_pct is None else f"{speedup_pct:.2f}"
        ql = "" if quality_loss_pct is None else f"{quality_loss_pct:.2f}"
        w.writerow([stage, name, str(base_ver), f"{latency_ms:.3f}", sp,
                    f"{quality:.6f}", ql, status, description])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase-3 状态机 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    si = sub.add_parser("init", help="从 phase2 backbone 初始化")
    si.add_argument("--run-dir", type=Path, required=True)
    si.add_argument("--tag", required=True)

    sr = sub.add_parser("resume", help="打印下一步动作 JSON")
    sr.add_argument("--run-dir", type=Path, required=True)

    sg = sub.add_parser("gate", help="双门判定,打印 {accept: bool}")
    sg.add_argument("--base-lat", type=float, required=True)
    sg.add_argument("--cand-lat", type=float, required=True)
    sg.add_argument("--baseline-q", type=float, required=True)
    sg.add_argument("--cand-q", type=float, required=True)

    sd = sub.add_parser("dispatch", help="按空闲卡派发,打印分配 JSON")
    sd.add_argument("--experiments", required=True, help="JSON 数组字符串")
    sd.add_argument("--free-gpus", required=True, help="逗号分隔 GPU id,如 0,1,2")

    sb = sub.add_parser("base-get", help="打印当前 base JSON")
    sb.add_argument("--run-dir", type=Path, required=True)

    args = ap.parse_args(argv)

    if args.cmd == "init":
        state = init_state(args.run_dir, args.tag)
        print(json.dumps(state["baseline"], ensure_ascii=False))
        return 0
    if args.cmd == "resume":
        print(json.dumps(next_action(load_state(args.run_dir)), ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        ok = passes_gate(args.base_lat, args.cand_lat, args.baseline_q, args.cand_q)
        print(json.dumps({"accept": ok}))
        return 0
    if args.cmd == "dispatch":
        exps = json.loads(args.experiments)
        gpus = [int(x) for x in args.free_gpus.split(",") if x != ""]
        print(json.dumps(plan_dispatch(exps, gpus), ensure_ascii=False))
        return 0
    if args.cmd == "base-get":
        print(json.dumps(base_get(load_state(args.run_dir) or {}) or {}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_phase3_state.py -v`
Expected: PASS(27 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/phase3_state.py tests/test_phase3_state.py
git commit -m "feat(phase3): results.tsv logging + state-machine CLI"
```

---

## Task 9: benchmark.py — adapter 加载 + 固定子集选取

**Files:**
- Create: `scripts/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'scripts.benchmark'`

- [ ] **Step 3: 写最小实现**

```python
# scripts/benchmark.py
"""冻结测速协议:动态加载 adapter,对 bench_config 钉死的固定子集做 batch=1 单条计时,
一次执行同产 speed.json(单条延迟均值/p50/p99 + 吞吐)与 predictions.jsonl(phase1 schema)。
本脚本不含任何框架/任务语义——那些都在 Claude 写的 adapter.py 里。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


def load_adapter(path: Path):
    """从路径动态加载 adapter 模块;须暴露 load_model() 与 infer_one()。"""
    spec = importlib.util.spec_from_file_location("p3_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_model") or not hasattr(mod, "infer_one"):
        raise ValueError("adapter 必须定义 load_model() 与 infer_one()")
    return mod


def _record_id(s: dict) -> str:
    return s.get("image_id", s.get("id"))


def select_records(bench_config: dict, dataset_dir: Path) -> list[dict]:
    """固定记录子集:有 subset_ids 按其顺序取;否则按 id 排序取前 n_records(默认全量)。"""
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    samples = meta["samples"]
    ids = bench_config.get("subset_ids")
    if ids:
        by = {_record_id(s): s for s in samples}
        return [by[i] for i in ids]
    n = int(bench_config.get("n_records", len(samples)))
    return sorted(samples, key=_record_id)[:n]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/benchmark.py tests/test_benchmark.py
git commit -m "feat(phase3): benchmark adapter loading + fixed-subset selection"
```

---

## Task 10: benchmark.py — 测速循环 + 双产出

**Files:**
- Modify: `scripts/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_benchmark.py::test_measure_produces_speed_and_predictions -v`
Expected: FAIL —— `AttributeError: module 'scripts.benchmark' has no attribute 'measure'`

- [ ] **Step 3: 写最小实现**(追加)

```python
def _sync() -> None:
    """有 GPU 时同步以测准内核耗时;无 torch/GPU(单测)时静默跳过。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _percentile(sorted_vals: list[float], q: float) -> float:
    """已排序列表的分位(最近秩)。q ∈ [0,1]。"""
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def measure(adapter_mod, dataset_dir: Path, bench_config: dict) -> tuple[dict, list[dict]]:
    """load 一次 → warmup N 条(丢弃)→ 对固定子集 batch=1 逐条计时 iters 轮。
    返回 (speed_dict, predictions_list);predictions 取自首个计时轮。"""
    handle = adapter_mod.load_model(bench_config.get("model_config", {}))
    records = select_records(bench_config, dataset_dir)
    warmup = int(bench_config.get("warmup", 3))
    iters = int(bench_config.get("iters", 5))

    for i in range(warmup):
        adapter_mod.infer_one(handle, records[i % len(records)])
    _sync()

    latencies: list[float] = []
    predictions: list[dict] = []
    for it in range(iters):
        for r in records:
            _sync()
            t0 = time.perf_counter()
            out = adapter_mod.infer_one(handle, r)
            _sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            if it == 0:
                predictions.append(out)

    latencies.sort()
    mean = sum(latencies) / len(latencies)
    speed = {
        "latency_mean_ms": mean,
        "p50_ms": _percentile(latencies, 0.50),
        "p99_ms": _percentile(latencies, 0.99),
        "throughput_qps": (1000.0 / mean) if mean > 0 else 0.0,
        "n_records": len(records),
        "warmup": warmup,
        "iters": iters,
        "framework": bench_config.get("framework", ""),
        "gpu_name": bench_config.get("gpu_name", ""),
        "gpu_count": int(bench_config.get("gpu_count", 1)),
    }
    return speed, predictions


def run(adapter_path, bench_config_path, dataset_dir, out, predictions_out) -> dict:
    bench_config = json.loads(Path(bench_config_path).read_text())
    mod = load_adapter(Path(adapter_path))
    speed, preds = measure(mod, Path(dataset_dir), bench_config)
    Path(out).write_text(json.dumps(speed, ensure_ascii=False, indent=2))
    with Path(predictions_out).open("w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return speed
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS(5 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/benchmark.py tests/test_benchmark.py
git commit -m "feat(phase3): benchmark timing loop + speed/predictions dual output"
```

---

## Task 11: benchmark.py — CLI

**Files:**
- Modify: `scripts/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: 写失败测试**

```python
import subprocess
import sys


def test_cli_run(bench_run):
    base, ds, adapter = bench_run
    cfg_path = base / "bench_config.json"
    cfg_path.write_text(json.dumps({"model_config": {"delay": 0.001},
                                    "warmup": 1, "iters": 1, "n_records": 2}))
    out = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark",
         "--adapter", str(adapter), "--bench-config", str(cfg_path),
         "--dataset", str(ds), "--out", str(base / "speed.json"),
         "--predictions", str(base / "predictions.jsonl")],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert out.returncode == 0
    assert (base / "speed.json").exists()
    assert len((base / "predictions.jsonl").read_text().splitlines()) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_benchmark.py::test_cli_run -v`
Expected: FAIL —— `returncode != 0`(`__main__` 无 CLI)

- [ ] **Step 3: 写最小实现**(追加)

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="冻结测速协议:产 speed.json + predictions.jsonl")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--bench-config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args(argv)
    speed = run(args.adapter, args.bench_config, args.dataset, args.out, args.predictions)
    print(json.dumps({"latency_mean_ms": speed["latency_mean_ms"],
                      "p50_ms": speed["p50_ms"], "p99_ms": speed["p99_ms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add scripts/benchmark.py tests/test_benchmark.py
git commit -m "feat(phase3): benchmark CLI entrypoint"
```

---

## Task 12: directions_schema — 扩展 phase3 tier

**Files:**
- Modify: `scripts/directions_schema.py`
- Test: `tests/test_directions_schema.py`

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_directions_schema.py`)

```python
from scripts import directions_schema as ds


def test_phase3_tiers_accepted():
    dirs = [{"slot": "a", "title": "FP8", "source_urls": ["u1"], "idea": "quant",
             "tier": "quantization", "needs_training": False}]
    assert ds.validate(dirs, ds.PHASE3_TIERS) == []


def test_phase3_tier_rejected_under_phase2_set():
    dirs = [{"slot": "a", "title": "FP8", "source_urls": ["u1"], "idea": "quant",
             "tier": "quantization", "needs_training": False}]
    errs = ds.validate(dirs, ds.PHASE2_TIERS)
    assert any("tier must be one of" in e for e in errs)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_directions_schema.py::test_phase3_tiers_accepted -v`
Expected: FAIL —— `AttributeError: module 'scripts.directions_schema' has no attribute 'PHASE3_TIERS'`

- [ ] **Step 3: 写最小实现**

修改 `scripts/directions_schema.py`:把固定的 `TIERS` 拆成两套并让 `validate` 接收可选 `tiers` 参数;CLI 增 `--tiers` 选择(默认 phase2,向后兼容)。

替换原第 11 行 `TIERS = {...}` 段为:

```python
PHASE2_TIERS = {"config", "post-process", "pipeline", "train", "infer-tune"}
PHASE3_TIERS = {"framework", "quantization", "kernel", "decoding", "compile", "parallel"}
TIERS = PHASE2_TIERS                       # 默认与历史行为一致
TIER_SETS = {"phase2": PHASE2_TIERS, "phase3": PHASE3_TIERS,
             "all": PHASE2_TIERS | PHASE3_TIERS}
```

把 `validate` 签名与 tier 校验改为接收参数:

```python
def validate(directions, tiers=TIERS) -> list[str]:
    if not isinstance(directions, list) or not directions:
        return ["directions must be a non-empty list"]
    errs: list[str] = []
    slots: list = []
    for i, d in enumerate(directions):
        if not isinstance(d, dict):
            errs.append(f"[{i}] must be an object")
            continue
        for key, typ in REQUIRED.items():
            if key not in d:
                errs.append(f"[{i}] missing field: {key}")
            elif not isinstance(d[key], typ):
                errs.append(f"[{i}] field {key} must be {typ.__name__}")
        if d.get("tier") not in tiers:
            errs.append(f"[{i}] tier must be one of {sorted(tiers)}")
        if not all(isinstance(u, str) for u in d.get("source_urls", [])):
            errs.append(f"[{i}] source_urls must be list[str]")
        slots.append(d.get("slot"))
    if len(set(slots)) != len(slots):
        errs.append(f"duplicate slots: {slots}")
    return errs
```

在 `main` 里增 `--tiers` 并传入:

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="directions.json 校验")
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--tiers", choices=list(TIER_SETS), default="phase2")
    args = ap.parse_args(argv)
    errs = validate(json.loads(args.file.read_text()), TIER_SETS[args.tiers])
    if errs:
        for e in errs:
            print(e)
        return 1
    print("ok")
    return 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_directions_schema.py -v`
Expected: PASS(原有用例 + 2 新增全过;默认参数保证历史行为不变)

- [ ] **Step 5: 提交**

```bash
git add scripts/directions_schema.py tests/test_directions_schema.py
git commit -m "feat(phase3): extend directions_schema with phase3 tiers"
```

---

## Task 13: 契约测试 — 3a→3b→3c 端到端文件流转

**Files:**
- Create: `tests/test_contract_phase3.py`

- [ ] **Step 1: 写失败测试**

```python
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
    bench_cfg = {"warmup": 1, "iters": 2, "n_records": 4, "framework": "toy"}

    # 0. init + baseline(主干原生:慢=5ms,质量 0.40)
    state = p3.init_state(run, "toy-tag")
    base_lat, base_q = _bench_and_score(run, ds, run / "phase3" / "baseline",
                                        0.005, 0.40, bench_cfg)
    state = p3.set_baseline(run, state, base_lat, 1000.0 / base_lat, quality=base_q)
    assert p3.next_action(state)["action"] == "framework_select"

    # 3a. 两个框架:vllm(快=2ms,质量达标 0.399)、sglang(更快但质量崩 0.30)
    for name, delay, score in [("vllm", 0.002, 0.399), ("sglang", 0.001, 0.30)]:
        p3.framework_add(run, state, name)
        lat, q = _bench_and_score(run, ds, run / "phase3" / "frameworks" / name,
                                  delay, score, bench_cfg)
        p3.framework_record(run, state, name, lat, q, "ready")
    state = p3.select_base(run, state)
    assert state["base_framework"]["name"] == "vllm"        # sglang 质量不达标被排除
    assert state["sub_phase"] == "single-card-loop"

    # 3b. 一轮单卡:方案 a 在 base 上再快一截(1ms)且质量达标 0.398 → 过双门
    state = p3.open_round(run, state, [{"slot": "a", "tier": "quantization"}])
    lat, q = _bench_and_score(run, ds,
                              run / "phase3" / "single_card" / "rounds" / "r001" / "exp_a",
                              0.001, 0.398, bench_cfg)
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

    # 3c. 两个并行方案:tp2(2 卡更快)、pp2(2 卡较慢);均质量达标
    for name, delay, gpus in [("tp2", 0.0005, 2), ("pp2", 0.0008, 2)]:
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
```

- [ ] **Step 2: 跑测试确认失败**

先临时把 Step 1 末尾断言改窄或直接运行——此时实现已存在(Task 1–11),应直接 PASS。若 import 期失败按报错修正。先运行:

Run: `uv run pytest tests/test_contract_phase3.py -v`
Expected: 若全链实现就位则 PASS;否则按首个失败断言定位缺口。

- [ ] **Step 3: 修正(若有)**

契约测试是回归网,不引入新生产代码;若失败,多半是字段名/路径不一致——回到对应 Task 修正实现(如 `exp_dir` 拼法、`speedup_pct` 单位)。不要在测试里掩盖。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_contract_phase3.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add tests/test_contract_phase3.py
git commit -m "test(phase3): end-to-end 3a->3b->3c contract dry-run"
```

---

## Task 14: GPU 冒烟测试(默认 skip)

**Files:**
- Create: `tests/test_smoke_phase3_gpu.py`

- [ ] **Step 1: 写测试(带 gpu marker,无 GPU 自动 skip)**

```python
# tests/test_smoke_phase3_gpu.py
"""真实环境冒烟:需 GPU + 真框架,默认 skip。CI/无卡机器不跑。
手动:uv run pytest tests/test_smoke_phase3_gpu.py -m gpu --no-header"""
import shutil

import pytest

pytestmark = pytest.mark.gpu


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="需要可用 CUDA GPU")
def test_benchmark_measures_real_latency(tmp_path):
    """最小真实 adapter(torch matmul)走 benchmark.measure,产出非零延迟与 CUDA sync。"""
    import json
    from pathlib import Path
    from scripts import benchmark as bm

    ds = tmp_path / "dataset"
    ds.mkdir()
    samples = [{"image_id": f"s{i}"} for i in range(3)]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import torch\n"
        "def load_model(config):\n"
        "    return torch.randn(512, 512, device='cuda')\n"
        "def infer_one(handle, record):\n"
        "    y = handle @ handle\n"
        "    return {'image_id': record['image_id'], 'score': float(y.sum().item()) * 0 + 0.4}\n"
    )
    mod = bm.load_adapter(adapter)
    speed, preds = bm.measure(mod, ds, {"warmup": 2, "iters": 3, "n_records": 3,
                                        "gpu_name": "real", "gpu_count": 1})
    assert speed["latency_mean_ms"] > 0
    assert len(preds) == 3
```

- [ ] **Step 2: 确认默认被 skip**

Run: `uv run pytest tests/test_smoke_phase3_gpu.py -v`
Expected: SKIPPED(无 GPU)或 PASS(有 GPU)。确认不报 ERROR/FAIL。

> 注:若 `gpu` marker 未注册会有 warning。检查 `pyproject.toml` 的 `[tool.pytest.ini_options] markers`;phase1/2 已注册 `gpu` marker 则无需改动。如未注册,在该段追加 `markers = ["gpu: 需要真实 GPU 的冒烟测试"]`。

- [ ] **Step 3: 提交**

```bash
git add tests/test_smoke_phase3_gpu.py
git commit -m "test(phase3): gpu-marked benchmark smoke (skipped by default)"
```

---

## Task 15: SKILL.md

**Files:**
- Create: `skills/autoexplore-phase3/SKILL.md`

- [ ] **Step 1: 写 SKILL.md**(镜像 phase2 SKILL 的 frontmatter + 结构)

```markdown
---
name: autoexplore-phase3
description: Use after phase-2 produced a winning backbone — drives phase-3 inference-speed optimization to production: 3a pick the fastest quality-passing single-card acceleration framework, 3b run a dual-gated single-card scheme loop (search GitHub/Arxiv/HF/paperswithcode → 3 directions → benchmark → accept on latency −10% AND quality loss ≤1%, compounding onto the base), then auto-advance on saturation to 3c multi-card parallelism, picking the fastest quality-passing scheme as the final production solution.
---

# autoexplore 第三阶段:模型推理速度优化

输入:第二阶段在 `runs/<tag>/phase2/state.json` 产出的获胜 `backbone`(权重 + 质量分);第一阶段冻结的 `dataset/`、`evaluate.py` 仍在位。
产出:生产级最终多卡推理方案(`phase3/state.json` 的 `final`),全程质量损失 ≤1%。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡;**复用主干 docker 环境**;按空闲度选卡。

**这是通用 skill,不绑定任何具体任务/数据集/模型。** 只依赖前阶段冻结的文件契约
(`evaluate.py` 的 CLI、`phase2/state.json` 的 backbone、`metrics.json` 的 `{primary_metric, metrics{}}`),不依赖任务语义。

## 两个冻结地基(全程不可变,保证可比)

- 质量:沿用 `evaluate.py` + `baseline/metrics.json`(= phase2 主干分)。
- 速度:`scripts/benchmark.py`(冻结测速协议)+ `phase3/bench_config.json`(本 run 钉死:固定子集/warmup/iters/卡型)+ `baseline/speed.json`。
- **唯一可变 = `adapter.py`**:Claude 按每个框架/方案写,暴露 `load_model()` / `infer_one(record)`。`benchmark.py` 一次执行同产 `speed.json` 与 `predictions.jsonl` → 喂 `evaluate.py` 出 `metrics.json`。同一次执行同时测速测质,杜绝漂移。

## 速度主指标与双门

- 主指标 = **单条延迟**(batch=1,warmup 后均值);吞吐为辅记录。
- **双门(贯穿全程)**:延迟相对**当前 base** 降 ≥10% **且** 质量相对**最初基线**损失 ≤1%。
- 3a/3c 选型:质量达标候选里**最小延迟者**胜(非"提升 ≥X")。

## 流程(需求文档第三阶段)

唯一真相源 = `runs/<tag>/phase3/state.json`,只经 `scripts/phase3_state.py` 读写。任意时刻先 `resume` 决定续点:
```bash
uv run scripts/phase3_state.py resume --run-dir runs/<tag>
# → {"action": "init|baseline|framework_select|search|execute|gate_check|parallel|done"}
```

### 步骤 0:Setup(唯一人工关卡)
续用 phase2 `<tag>`;worktree 切 `speedup/<tag>`(共享目录用 worktree,不动共享 HEAD);
钉 `bench_config.json`;在主干原 phase2 镜像上跑 benchmark+evaluate 建 baseline;
`phase3_state init` → `sub_phase=framework-select`。确认后进入自主流程,之后不再逐步问人。

### 模式 3a:单卡基础框架选型(一次性,按 phase1 复现范式)
GitHub 搜本模型领域推理加速框架 → 选 ≤3 排序 → 逐个复现(已有该模型直接接 adapter;
没有则参照相似模型改造)→ benchmark+evaluate → **质量达标里最小延迟者 = base v0**;
无任何达标则退回主干原生推理作 base。进 3b。

### 模式 3b:单卡加速方案优化循环(饱和自动进 3c,可人工触发)
LOOP:搜 github/Arxiv/HF/paperswithcode 选 3 个可复用加速方向(量化/KV-cache/投机解码/
kernel 融合/CUDA graph/torch.compile…,`directions.json` 用 `--tiers phase3` 校验、跨轮去重)→
`dispatch` 按空闲卡并发、训练型(量化校准/草稿训练)排队 → 各在当前 base 上叠加实现 adapter →
benchmark+evaluate 评分 → `gate` 判双门 → 过门 `promote_base`+commit、`dry_streak` 清零、回搜索;
无过门 `bump_dry_streak`;`saturation_check` 到 K 轮 → 进 3c。**绝不停下问人**。

### 模式 3c:多卡并行扩展(一次性,≤3 SOTA 方案)
取最终单卡 base → 选 ≤3 并行方案(TP/PP/EP/SP/replica)→ 逐个多卡实现 adapter →
benchmark(同测单条延迟)+ 质量校验 → **质量达标里最小延迟者 = final** → `sub_phase=done`。交付。

完整细则见 [references/speedup-loop.md](references/speedup-loop.md)。

## 关键纪律
- `dataset/`+`evaluate.py`+`benchmark.py`+`bench_config.json` 全程**不可变**,保证可比。
- **双门 AND**:提速 ≥10%(vs 当前 base)且 质量损失 ≤1%(vs 最初基线,防累积击穿)。
- **keep/discard 用 state.json 指针,不用 git reset**:失败框架/方案目录保留作研究档案。
- 容器/训练输出进 log,只失败时 `tail`;逐实验重试上限 3,crash 不阻塞同轮兄弟。
- 中断可恢复:入口 `resume` 读 state.json;已 scored slot 跳过、已晋升 base 不回退、按 sub_phase 续。
- **容器纪律(继承前阶段)**:每次 `docker run` 带 `--user $UID:$GID --runtime=nvidia`;
  caches 以 `:ro` 挂 `/cache/{modelscope,huggingface,torch}`,env 注入对应 `*_CACHE`/`*_HOME`。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/phase3_state.py {init,resume,gate,dispatch,base-get}` | 状态/双门/派发(确定性核心) |
| `scripts/benchmark.py` | 冻结测速协议:产 speed.json + predictions.jsonl |
| `scripts/directions_schema.py --file <f> --tiers phase3` | 方向 schema 校验(phase3 tier) |
| `scripts/train_launch.py` | 量化校准/草稿训练:数据出处/多卡启动/ckpt 续/预算 |
| `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py` | 复用前阶段 |
```

- [ ] **Step 2: 校验 frontmatter 可被解析**

Run: `head -3 skills/autoexplore-phase3/SKILL.md`
Expected: 首行 `---`、`name: autoexplore-phase3`、`description:` 起头。

- [ ] **Step 3: 提交**

```bash
git add skills/autoexplore-phase3/SKILL.md
git commit -m "docs(phase3): add autoexplore-phase3 SKILL.md"
```

---

## Task 16: references/speedup-loop.md

**Files:**
- Create: `skills/autoexplore-phase3/references/speedup-loop.md`

- [ ] **Step 1: 写细则文档**(镜像 phase2 `optimization-loop.md` 风格,给出可复制命令)

```markdown
# 推理速度优化循环细则(第三阶段)

承接 SKILL.md。所有路径相对 repo 根;命令均从 repo 根 `uv run`。
状态唯一真相源 = `runs/<tag>/phase3/state.json`,只经 `scripts/phase3_state.py` 读写。
进入任意阶段前先 `resume`:
```bash
uv run scripts/phase3_state.py resume --run-dir runs/<tag>
# → {"action": "init|baseline|framework_select|search|execute|gate_check|parallel|done"}
```

## 步骤 0:Setup(唯一人工关卡)
1. 续用 phase2 `<tag>`;`git worktree add .claude/worktrees/speedup-<tag> -b speedup/<tag> HEAD`(共享目录纪律)。
2. 钉 `runs/<tag>/phase3/bench_config.json`,例:
   ```json
   {"subset_ids": ["<固定挑选的代表性样本 id>"], "warmup": 5, "iters": 20,
    "gpu_name": "<目标卡型>", "gpu_count": 1, "framework": "baseline",
    "model_config": {"<主干推理所需参数>": "..."}}
   ```
   子集要固定且有代表性;warmup/iters 取够稳(p99 不抖)。**钉死后不再改**。
3. 建 baseline(主干原生推理,原 phase2 镜像):写 `baseline/adapter.py` → 测速+评分:
   ```bash
   uv run scripts/benchmark.py --adapter runs/<tag>/phase3/baseline/adapter.py \
     --bench-config runs/<tag>/phase3/bench_config.json --dataset runs/<tag>/dataset \
     --out runs/<tag>/phase3/baseline/speed.json \
     --predictions runs/<tag>/phase3/baseline/predictions.jsonl
   uv run scripts/compute_metrics.py --evaluate-py runs/<tag>/evaluate.py \
     --predictions runs/<tag>/phase3/baseline/predictions.jsonl \
     --dataset runs/<tag>/dataset --out runs/<tag>/phase3/baseline/metrics.json
   ```
4. `uv run scripts/phase3_state.py init --run-dir runs/<tag> --tag <tag>`,再用
   `baseline/speed.json` 的 `latency_mean_ms`/`throughput_qps` 与 `baseline/metrics.json` 的
   `primary_metric` 经 `set_baseline` 写入。确认后进入自主流程。

## 模式 3a:单卡基础框架选型
1. GitHub 搜本模型领域推理加速框架(WebSearch/WebFetch),按官方披露加速比排序,选 ≤3。
2. 逐框架(`frameworks/<fw>/`):
   - `docker_env.py build` 出框架镜像(官方镜像或 `FROM` 派生);
   - 框架已有该模型 → adapter 薄包官方 API;没有 → 参照相似模型改造 adapter;
   - benchmark + compute_metrics;`framework_record` 记 latency/quality/status;
   - 失败读 `build.log`/`run.log` 改镜像或 adapter,重试上限 3,仍败记 crash 换下一个。
3. 全部记完 → `select_base`:质量达标里最小延迟者 = base v0;无则退回主干原生。进 3b。

## 模式 3b:单卡加速方案优化循环(LOOP,饱和自动进 3c)
### a. 搜方向
WebSearch/WebFetch 在 github/Arxiv/HF/paperswithcode 搜**跨领域可复用**加速方向,选 3,
写 `single_card/rounds/rNNN/directions.json`(tier ∈ phase3 集 + `expected_speedup`/`quality_risk`),
校验并去重:
```bash
uv run scripts/directions_schema.py --file <directions.json> --tiers phase3
```
跨轮去重:用 `directions_seen` 跳过已试方向(标题+URL 指纹)。
### b. 派发与实现
```bash
uv run scripts/phase3_state.py dispatch \
  --experiments '[{"slot":"a","needs_gpus":1,"is_training":false}, ...]' --free-gpus 0,1,2
```
各 `exp_*/` 在**当前 base** 上叠加实现 `adapter.py`(需校准/草稿训练用 `train_launch.py` 产物);
benchmark + compute_metrics;`open_round`/`record_slot` 记分(自动算 speedup% vs base、loss% vs 基线)。
### c. 双门与晋升
对最佳 slot:
```bash
uv run scripts/phase3_state.py gate --base-lat <当前base延迟> --cand-lat <候选延迟> \
  --baseline-q <最初基线质量> --cand-q <候选质量>     # → {"accept": bool}
```
- accept → `promote_base`(version_n++、`dry_streak` 清零)+ `git commit` 留痕 → 回 a;
- 无过门 → `append_phase3_result` 记 discard/crash + `bump_dry_streak`。
### d. 饱和
```bash
# dry_streak ≥ saturation_k(默认 3)→ 自动进多卡;也可人工 force
```
`saturation_check`(force 可人工提前)推进 `sub_phase=multi-card`。**绝不停下问人**;
"没主意"时重读论文、组合近似命中、试更激进改动(对齐 autoresearch 纪律)。

## 模式 3c:多卡并行扩展
1. 取最终单卡 base;选 ≤3 SOTA 并行方案(TP/PP/EP/SP/replica,框架而定)。
2. 逐方案(`multi_card/scheme_<name>/`):多卡实现 adapter(`dispatch` 要够 N 卡),
   benchmark(同测单条延迟主指标)+ compute_metrics;`parallel_add`/`parallel_record`。
3. `select_final`:质量达标里最小延迟者 = final(`gpu_count` 随之),无则回落单卡 base。
   `sub_phase=done`。交付最终生产级多卡推理方案。

## 错误处理速查
- 镜像/依赖构建失败:retry≤3,读 `build.log` 改派生 Dockerfile,仍败记 crash。
- 框架无该模型:参照相似模型改 adapter;无可参照记 crash 标因。
- 框架数值改动致质量超损:`passes_quality=false`,排除出 base 候选。
- 测速抖动:加大 warmup/iters,固定子集复测。
- 卡不足:训练型/多卡排队,非训练优先。
- 中断:`resume` 按 sub_phase 续,已 scored slot 跳过、已晋升 base 不回退。
```

- [ ] **Step 2: 校验 SKILL 引用路径成立**

Run: `ls skills/autoexplore-phase3/references/speedup-loop.md`
Expected: 文件存在(SKILL.md 的相对链接可达)。

- [ ] **Step 3: 全量回归**

Run: `uv run pytest tests/ -q`
Expected: 全绿(phase1/2 既有用例 + phase3 新增);gpu 用例 SKIPPED。

- [ ] **Step 4: 提交**

```bash
git add skills/autoexplore-phase3/references/speedup-loop.md
git commit -m "docs(phase3): add speedup-loop reference"
```

---

## 完成标准(Definition of Done)

- [ ] `scripts/phase3_state.py` + `scripts/benchmark.py` 实现完整,单测全绿。
- [ ] `directions_schema.py` 扩展 phase3 tier 且 phase2 行为不变(默认参数)。
- [ ] 契约测试覆盖 3a→3b→3c 全链文件流转,绿。
- [ ] GPU 冒烟默认 skip,不报错。
- [ ] `skills/autoexplore-phase3/{SKILL.md,references/speedup-loop.md}` 就位,内部链接可达。
- [ ] `uv run pytest tests/ -q` 全绿。
- [ ] 分支 `feature/phase3-speedup` 上逐 Task 提交;完成后按 `GIT_CONVENTIONS.md` squash 合回。
