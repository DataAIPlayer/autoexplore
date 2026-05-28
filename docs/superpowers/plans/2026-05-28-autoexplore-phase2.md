# autoexplore 第二阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建第二阶段「模型效果迭代优化」agent(独立第二个 skill),覆盖需求文档第二阶段:以阶段一选中的模型为可演进的模型主干,诊断短板 → 便宜的推理管道调优闸门 → 不终止的搜索优化循环(相对 +5% 晋升新主干)。

**Architecture:** 方案 C(混合):一个薄而受测的确定性核心 `phase2_state.py` 管最易错的记账(主干指针 / 轮次 / 实验登记 / 晋升门数学 / 空闲卡并发派发 / 中断恢复);`diagnose.py` 把冻结 `evaluate.py` 当黑盒在子集视图上跑出短板分解;`directions_schema.py` 校验 agent 搜出的方向;`train_launch.py` 是通用训练壳。判断密集步骤(读诊断、搜方向、写实现代码)由 SKILL.md + reference 引导 agent 完成。脚本通过文件系统约定(`runs/<tag>/phase2/` 工作目录 + JSON schema)通信,可在无 Docker/GPU 的开发机上用玩具 `evaluate.py` 和 mock 单测。

**Tech Stack:** Python 3.12,uv,pytest(沿用阶段一);phase-2 脚本仅用标准库(subprocess/json/csv/argparse/pathlib/dataclasses/tempfile/os/time);复用阶段一 `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py`;运行时依赖 docker CLI + nvidia-smi(仅服务器,冒烟测试用)。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `scripts/phase2_state.py` | 确定性受测核心:state.json 读写、主干初始化/晋升、轮次与实验登记、晋升门(相对 +5%)、空闲卡派发、中断恢复决策、CLI |
| `scripts/diagnose.py` | 短板诊断:把冻结 `evaluate.py` 当黑盒,在 dataset 子集视图(全集/逐样本/分组)上反复调用,产出 `diag_<id>.{json,md}` |
| `scripts/directions_schema.py` | 薄校验:`directions.json` 结构与 tier 枚举 |
| `scripts/train_launch.py` | 通用训练壳:数据出处记录、多卡启动命令拼装、checkpoint 续选、预算执行(超时 kill) |
| `tests/conftest.py` | 共享 `toy_run` fixture:在 tmp 目录造带 results.tsv/dataset/玩具 evaluate.py 的假 run |
| `tests/helpers.py` | 测试用纯函数 `write_predictions`(正常 import,不放进 conftest) |
| `tests/test_phase2_state.py` | phase2_state 单测(晋升门边界、init、派发训练排队、轮次往返、resume、原子写、方向去重) |
| `tests/test_diagnose.py` | diagnose 单测(子集视图、黑盒调用、worst-K、分组、无分类字段降级) |
| `tests/test_directions_schema.py` | directions_schema 单测(合法/非法样例) |
| `tests/test_train_launch.py` | train_launch 单测(ckpt 续选、出处记录、命令拼装、预算超时 kill) |
| `tests/test_contract_phase2.py` | 端到端 dry-run 契约测试:diag → directions → exp metrics → record → promote → results.tsv/state.json 全链路 schema |
| `tests/test_smoke_phase2_gpu.py` | 真实 Docker 冒烟(`@pytest.mark.gpu`,默认 skip) |
| `skills/autoexplore-phase2/SKILL.md` | 第二阶段流程与决策编排(独立 skill 入口) |
| `skills/autoexplore-phase2/references/optimization-loop.md` | 优化循环细则(从 SKILL.md 引用) |

约定:phase-2 脚本是独立 CLI(从 repo 根 `uv run scripts/<name>.py ...`),也可作为模块 import 供测试调用。phase-1 根布局(根 `SKILL.md` + `references/reproduction-loop.md`)不动;phase-2 自包含在 `skills/autoexplore-phase2/`,共享根 `scripts/`(命令均从 repo 根运行)。最终插件化打包(把两个 skill 都收进 `skills/`)是后续打包步骤,本计划不做。

---

## Task 1: 共享测试夹具 `toy_run` + helpers

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/helpers.py`

构造一个无 Docker/GPU 依赖的假 run,被后续多个测试复用(DRY)。玩具 `evaluate.py` 只用标准库,且遵守冻结契约 `--predictions --dataset --out` → 输出 `{primary_metric, metrics}`;它从每条 prediction 的 `score` 字段取分、对 dataset metadata 列出的样本求均值,使子集视图的得分可预期(便于验证 worst-K / 分组)。

> 说明:phase-1 测试不用 conftest、各文件内联造 run。phase-2 有 4 个新测试文件共享同一玩具 run,故引入一个 `toy_run` 夹具(pytest 标准做法,conftest 中自动发现、无需 import);纯函数 `write_predictions` 放 `tests/helpers.py` 正常 import,避免“从 conftest import”的坏味道。phase-1 既有测试不动。

- [ ] **Step 1: 写 tests/helpers.py(纯函数,正常 import)**

```python
"""测试用纯函数(正常 import,勿放进 conftest 以免 import 坏味道)。"""
import json
from pathlib import Path


def write_predictions(path: Path, id_scores: dict):
    """按 {image_id: score} 写 predictions.jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sid, sc in id_scores.items():
            f.write(json.dumps({"image_id": sid, "score": sc, "layers": []}) + "\n")
```

- [ ] **Step 2: 写 tests/conftest.py(toy_run 夹具,自包含)**

```python
"""共享测试夹具:造一个无 Docker/GPU 依赖的玩具 run。"""
import json

import pytest

# 玩具 evaluate.py:遵守冻结 CLI 契约,从 prediction 的 score 字段取分求均值。
TOY_EVALUATE = '''\
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    preds = {}
    with a.predictions.open() as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                preds[o["image_id"]] = o
    meta = json.loads((a.dataset / "metadata.json").read_text())
    ids = [s["image_id"] for s in meta["samples"]]
    scores, n_skipped = [], 0
    for sid in ids:
        p = preds.get(sid)
        if p is None:
            n_skipped += 1
            scores.append(0.0)
        else:
            scores.append(float(p.get("score", 0.0)))
    primary = sum(scores) / len(scores) if scores else 0.0
    out = {"primary_metric": primary,
           "metrics": {"score_mean": primary, "n_samples": len(ids) - n_skipped,
                       "n_skipped": n_skipped}}
    a.out.write_text(json.dumps(out))
    print(f"primary_metric={primary:.6f}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def toy_run(tmp_path):
    """返回 run_dir。含 results.tsv(两个 ready 模型)、dataset(3 样本,带 group 字段)、玩具 evaluate.py。"""
    run = tmp_path / "runs" / "toy-tag"
    ds = run / "dataset"
    ds.mkdir(parents=True)
    samples = [
        {"image_id": "s1", "group": "A"},
        {"image_id": "s2", "group": "A"},
        {"image_id": "s3", "group": "B"},
    ]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    for s in samples:
        d = ds / s["image_id"]
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"image_id": s["image_id"]}))
    (run / "evaluate.py").write_text(TOY_EVALUATE)
    (run / "results.tsv").write_text(
        "model\tprimary_metric\tmemory_gb\tstatus\tdescription\n"
        "toy-a\t0.400000\t10.0\tready\tbest baseline\n"
        "toy-b\t0.300000\t8.0\tready\trunner up\n"
        "toy-c\t0.000000\t0.0\tcrash\tfailed\n"
    )
    return run
```

- [ ] **Step 3: 验证夹具与 helpers 可被发现/导入**

Run: `uv run pytest --fixtures -q 2>/dev/null | grep -q toy_run && python -c "from tests.helpers import write_predictions; print('ok')"`
Expected: 打印 `ok`(toy_run 夹具被发现,helpers 正常 import)。

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/helpers.py
git commit -m "test(phase2): add shared toy_run fixture and write_predictions helper"
```

---

## Task 2: phase2_state.py — 纯函数 + 状态 IO + 初始化 + 主干 + 方向去重

**Files:**
- Create: `scripts/phase2_state.py`
- Test: `tests/test_phase2_state.py`

- [ ] **Step 1: 写失败测试(纯函数 + init + 主干 + 方向)**

```python
import json
from pathlib import Path

import pytest

from scripts import phase2_state as ps


def test_is_significant_relative_5pct():
    assert ps.is_significant(0.42, 0.40) is True      # +5.0% 恰好达门
    assert ps.is_significant(0.4199, 0.40) is False    # +4.97% 不够
    assert ps.is_significant(0.50, 0.40) is True
    assert ps.is_significant(0.40, 0.40) is False      # 持平不晋升


def test_is_significant_zero_backbone():
    assert ps.is_significant(0.01, 0.0) is True
    assert ps.is_significant(0.0, 0.0) is False


def test_best_ready_from_results_picks_highest_ready(toy_run):
    model, metric = ps.best_ready_from_results(toy_run / "results.tsv")
    assert model == "toy-a"
    assert metric == pytest.approx(0.40)


def test_best_ready_raises_when_no_ready(tmp_path):
    f = tmp_path / "results.tsv"
    f.write_text("model\tprimary_metric\tmemory_gb\tstatus\tdescription\n"
                 "x\t0.0\t0.0\tcrash\tnope\n")
    with pytest.raises(ValueError):
        ps.best_ready_from_results(f)


def test_init_state_sets_backbone_from_best_ready(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    assert state["backbone"]["id"] == "p1:toy-a"
    assert state["backbone"]["primary_metric"] == pytest.approx(0.40)
    assert state["backbone"]["version_n"] == 0
    assert state["inference_tuning"] == "pending"
    assert state["last_diagnosed_version"] == -1
    # 落盘且可重新加载
    loaded = ps.load_state(toy_run)
    assert loaded["backbone"]["id"] == "p1:toy-a"


def test_promote_backbone_bumps_version_and_resets_tuning(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    ps.mark_diagnosed(toy_run, state)
    ps.set_inference_tuning(toy_run, state, "explored")
    state = ps.promote_backbone(toy_run, state, "r001:exp_a",
                                "rounds/r001/exp_a", 0.50, {"score_mean": 0.50})
    assert state["backbone"]["version_n"] == 1
    assert state["backbone"]["id"] == "r001:exp_a"
    assert state["backbone"]["primary_metric"] == pytest.approx(0.50)
    assert state["inference_tuning"] == "pending"      # 新主干重评便宜调优
    assert len(state["backbone_history"]) == 2


def test_directions_dedup(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    d = {"title": "Foo Method", "source_urls": ["http://arxiv.org/abs/1"]}
    assert ps.directions_seen(state, d) is False
    ps.directions_add(toy_run, state, [d])
    assert ps.directions_seen(state, d) is True
    # 顺序无关的指纹
    d2 = {"title": "Foo Method", "source_urls": ["http://arxiv.org/abs/1"]}
    assert ps.directions_seen(state, d2) is True


def test_save_state_is_atomic(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    state["round_counter"] = 7
    ps.save_state(toy_run, state)
    assert not (toy_run / "phase2" / "state.json.tmp").exists()  # 临时文件已替换
    assert ps.load_state(toy_run)["round_counter"] == 7
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_phase2_state.py -q`
Expected: FAIL,`ModuleNotFoundError` 或 `AttributeError: module 'scripts.phase2_state' has no attribute ...`

- [ ] **Step 3: 实现 phase2_state.py(本任务范围)**

```python
"""Phase-2 状态机 (state.json 唯一真相源):主干指针 / 轮次 / 实验登记 /
晋升门 / 空闲卡并发派发 / 中断恢复。本模块独占读写 state.json。"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROMOTE_REL = 0.05  # 相对 +5% 才晋升新主干


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def best_ready_from_results(results_tsv: Path) -> tuple[str, float]:
    """读 phase-1 results.tsv,返回 status=ready 行里 primary_metric 最高的 (model, metric)。
    无 ready 行抛 ValueError。"""
    best_model, best_metric = None, -1.0
    with results_tsv.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") != "ready":
                continue
            m = float(row["primary_metric"])
            if m > best_metric:
                best_model, best_metric = row["model"], m
    if best_model is None:
        raise ValueError(f"no status=ready rows in {results_tsv}")
    return best_model, best_metric


def is_significant(candidate: float, backbone: float, rel: float = PROMOTE_REL) -> bool:
    """候选是否相对当前主干显著提升 (>= backbone*(1+rel))。"""
    if backbone <= 0.0:
        return candidate > 0.0
    return candidate >= backbone * (1.0 + rel)


def _direction_fingerprint(d: dict) -> str:
    """方向去重指纹:title + 排序后的 source_urls,小写归一,顺序无关。"""
    urls = "|".join(sorted(d.get("source_urls", [])))
    return (d.get("title", "").strip() + "|" + urls).lower()


def load_state(run_dir: Path) -> dict | None:
    p = run_dir / "phase2" / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_state(run_dir: Path, state: dict) -> None:
    p = run_dir / "phase2" / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(p)  # 原子替换


def init_state(run_dir: Path, tag: str) -> dict:
    model, metric = best_ready_from_results(run_dir / "results.tsv")
    bb = {"id": f"p1:{model}", "source_dir": f"models/{model}",
          "primary_metric": metric, "metrics": {}, "version_n": 0}
    state = {
        "tag": tag,
        "backbone": bb,
        "backbone_history": [{"version_n": 0, "id": bb["id"],
                              "primary_metric": metric, "promoted_at": _now()}],
        "last_diagnosed_version": -1,
        "inference_tuning": "pending",
        "round_counter": 0,
        "rounds": [],
        "directions_tried": [],
        "updated_at": _now(),
    }
    save_state(run_dir, state)
    return state


def promote_backbone(run_dir: Path, state: dict, exp_id: str, source_dir: str,
                     primary_metric: float, metrics: dict) -> dict:
    v = state["backbone"]["version_n"] + 1
    state["backbone"] = {"id": exp_id, "source_dir": source_dir,
                         "primary_metric": primary_metric, "metrics": metrics,
                         "version_n": v}
    state["backbone_history"].append({"version_n": v, "id": exp_id,
                                      "primary_metric": primary_metric,
                                      "promoted_at": _now()})
    state["inference_tuning"] = "pending"  # 新主干重新评估便宜调优
    save_state(run_dir, state)
    return state


def mark_diagnosed(run_dir: Path, state: dict) -> dict:
    state["last_diagnosed_version"] = state["backbone"]["version_n"]
    save_state(run_dir, state)
    return state


def set_inference_tuning(run_dir: Path, state: dict, value: str) -> dict:
    state["inference_tuning"] = value  # pending|explored|applied|skipped
    save_state(run_dir, state)
    return state


def directions_seen(state: dict, direction: dict) -> bool:
    return _direction_fingerprint(direction) in state.get("directions_tried", [])


def directions_add(run_dir: Path, state: dict, directions: list[dict]) -> dict:
    for d in directions:
        fp = _direction_fingerprint(d)
        if fp not in state["directions_tried"]:
            state["directions_tried"].append(fp)
    save_state(run_dir, state)
    return state
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_phase2_state.py -q`
Expected: PASS(8 个测试全绿)

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2_state.py tests/test_phase2_state.py
git commit -m "feat(phase2): add state core — gate, backbone init/promote, directions dedup"
```

---

## Task 3: phase2_state.py — 晋升门派发 + 轮次登记 + resume 决策

**Files:**
- Modify: `scripts/phase2_state.py`
- Modify: `tests/test_phase2_state.py`

- [ ] **Step 1: 追加失败测试(派发 / 轮次 / resume)**

```python
def test_plan_dispatch_non_training_first_training_queues():
    exps = [
        {"slot": "a", "needs_gpus": 1, "is_training": True},
        {"slot": "b", "needs_gpus": 1, "is_training": False},
        {"slot": "c", "needs_gpus": 1, "is_training": False},
    ]
    plan = ps.plan_dispatch(exps, free_gpus=[0, 1])  # 只够两个非训练
    assert set(plan["assigned"].keys()) == {"b", "c"}  # 非训练优先拿卡
    assert plan["queued"] == ["a"]                      # 训练型排队


def test_plan_dispatch_multi_gpu_need():
    exps = [{"slot": "a", "needs_gpus": 3, "is_training": True},
            {"slot": "b", "needs_gpus": 1, "is_training": False}]
    plan = ps.plan_dispatch(exps, free_gpus=[0, 1])
    assert plan["assigned"] == {"b": [0]}   # a 需 3 卡但只剩 1,排队
    assert plan["queued"] == ["a"]


def test_round_open_and_record_lifecycle(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    dirs = [{"slot": "a", "tier": "config", "title": "t1", "source_urls": []},
            {"slot": "b", "tier": "train", "title": "t2", "source_urls": []},
            {"slot": "c", "tier": "pipeline", "title": "t3", "source_urls": []}]
    state = ps.open_round(toy_run, state, dirs)
    assert state["rounds"][-1]["id"] == "r001"
    assert state["rounds"][-1]["status"] == "open"
    assert len(state["directions_tried"]) == 3  # open 同时登记去重
    # 记录前两个 slot → running
    state = ps.record_slot(toy_run, state, "r001", "a", 0.50, "done")
    state = ps.record_slot(toy_run, state, "r001", "b", 0.0, "crash")
    assert state["rounds"][-1]["status"] == "running"
    # delta% 相对主干 0.40
    slot_a = next(s for s in state["rounds"][-1]["slots"] if s["slot"] == "a")
    assert slot_a["delta_pct"] == pytest.approx(25.0)
    # 记录最后一个 → scored
    state = ps.record_slot(toy_run, state, "r001", "c", 0.41, "done")
    assert state["rounds"][-1]["status"] == "scored"


def test_close_round(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    state = ps.open_round(toy_run, state, [{"slot": "a", "tier": "config",
                                            "title": "t", "source_urls": []}])
    state = ps.close_round(toy_run, state, "r001")
    assert state["rounds"][-1]["status"] == "done"


def test_next_action_sequence(toy_run):
    # 刚 init:未诊断 → diagnose
    state = ps.init_state(toy_run, "toy-tag")
    assert ps.next_action(state)["action"] == "diagnose"
    # 诊断后:便宜调优待定 → infer_tune
    ps.mark_diagnosed(toy_run, state)
    assert ps.next_action(state)["action"] == "infer_tune"
    # 调优探索完 → search(无 round)
    ps.set_inference_tuning(toy_run, state, "explored")
    assert ps.next_action(state)["action"] == "search"
    # 开了 round 未跑完 → execute
    state = ps.open_round(toy_run, state, [{"slot": "a", "tier": "config",
                                            "title": "t", "source_urls": []}])
    assert ps.next_action(state)["action"] == "execute"
    # 全部 scored → promote_check
    state = ps.record_slot(toy_run, state, "r001", "a", 0.41, "done")
    assert ps.next_action(state)["action"] == "promote_check"
    # 关闭 round → 回 search
    state = ps.close_round(toy_run, state, "r001")
    assert ps.next_action(state)["action"] == "search"


def test_next_action_init_when_no_backbone():
    assert ps.next_action({})["action"] == "init"


def test_append_phase2_result_schema(toy_run):
    ps.append_phase2_result(toy_run, "r001", "exp_a", 0, 0.45, 12.5, "keep", "config tweak")
    ps.append_phase2_result(toy_run, "r001", "exp_b", 0, 0.0, None, "crash", "OOM")
    import csv as _csv
    rows = list(_csv.reader((toy_run / "phase2" / "results.tsv").open(), delimiter="\t"))
    assert rows[0] == ["round", "exp", "base_backbone_ver", "primary_metric",
                       "delta_pct", "status", "description"]
    assert rows[1][0] == "r001" and rows[1][1] == "exp_a"
    assert rows[1][4] == "12.50" and rows[1][5] == "keep"
    assert rows[2][4] == "" and rows[2][5] == "crash"   # delta_pct=None → 空
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_phase2_state.py -q`
Expected: FAIL,缺 `plan_dispatch` / `open_round` / `record_slot` / `close_round` / `next_action` / `append_phase2_result`

- [ ] **Step 3: 追加实现到 phase2_state.py**

```python
def plan_dispatch(experiments: list[dict], free_gpus: list[int]) -> dict:
    """按空闲卡为就绪实验派发 GPU。非训练优先,训练型卡不够则排队。
    experiments: [{"slot","needs_gpus","is_training"}]; free_gpus: [gpu_id,...]。
    返回 {"assigned": {slot: [gpu,...]}, "queued": [slot,...]}。"""
    ordered = sorted(experiments, key=lambda e: (bool(e.get("is_training")), e["slot"]))
    pool = list(free_gpus)
    assigned: dict[str, list[int]] = {}
    queued: list[str] = []
    for e in ordered:
        need = int(e.get("needs_gpus", 1))
        if len(pool) >= need:
            assigned[e["slot"]] = [pool.pop(0) for _ in range(need)]
        else:
            queued.append(e["slot"])
    return {"assigned": assigned, "queued": queued}


def _find_round(state: dict, round_id: str) -> dict:
    for r in state["rounds"]:
        if r["id"] == round_id:
            return r
    raise KeyError(round_id)


def open_round(run_dir: Path, state: dict, directions: list[dict]) -> dict:
    state["round_counter"] += 1
    rid = f"r{state['round_counter']:03d}"
    slots = [{"slot": d["slot"], "exp_dir": f"rounds/{rid}/exp_{d['slot']}",
              "tier": d.get("tier", ""), "primary_metric": None,
              "delta_pct": None, "status": "pending"} for d in directions]
    state["rounds"].append({"id": rid, "status": "open", "slots": slots})
    directions_add(run_dir, state, directions)  # 已含 save_state
    return state


def record_slot(run_dir: Path, state: dict, round_id: str, slot: str,
                primary_metric: float, status: str) -> dict:
    """status ∈ {done, crash}。done 的算 delta% vs 当前主干。全 slot 终态则轮转 scored。"""
    rnd = _find_round(state, round_id)
    bb = state["backbone"]["primary_metric"]
    for s in rnd["slots"]:
        if s["slot"] == slot:
            s["primary_metric"] = primary_metric
            s["delta_pct"] = ((primary_metric - bb) / bb * 100.0) if bb > 0 else None
            s["status"] = status
    if all(s["status"] in ("done", "crash") for s in rnd["slots"]):
        rnd["status"] = "scored"
    else:
        rnd["status"] = "running"
    save_state(run_dir, state)
    return state


def close_round(run_dir: Path, state: dict, round_id: str) -> dict:
    _find_round(state, round_id)["status"] = "done"
    save_state(run_dir, state)
    return state


def next_action(state: dict) -> dict:
    """中断恢复决策:读 state 给出下一步动作。优先级见 spec §4 控制流。"""
    if not state.get("backbone"):
        return {"action": "init"}
    if state["backbone"]["version_n"] != state.get("last_diagnosed_version", -1):
        return {"action": "diagnose", "version_n": state["backbone"]["version_n"]}
    if state.get("inference_tuning") == "pending":
        return {"action": "infer_tune"}
    rounds = state.get("rounds", [])
    if not rounds or rounds[-1]["status"] == "done":
        return {"action": "search"}
    last = rounds[-1]
    if last["status"] == "scored":
        return {"action": "promote_check", "round": last["id"]}
    return {"action": "execute", "round": last["id"]}


PHASE2_RESULTS_HEADER = ["round", "exp", "base_backbone_ver", "primary_metric",
                         "delta_pct", "status", "description"]


def append_phase2_result(run_dir: Path, round_id: str, exp: str,
                         base_backbone_ver: int, primary_metric: float,
                         delta_pct: float | None, status: str,
                         description: str) -> None:
    """追加一行 phase2/results.tsv(7 列,扩展 phase-1 五列;append-only 研究日志)。"""
    path = run_dir / "phase2" / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if is_new:
            w.writerow(PHASE2_RESULTS_HEADER)
        dp = "" if delta_pct is None else f"{delta_pct:.2f}"
        w.writerow([round_id, exp, str(base_backbone_ver),
                    f"{primary_metric:.6f}", dp, status, description])
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_phase2_state.py -q`
Expected: PASS(全部 15 个测试绿)

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2_state.py tests/test_phase2_state.py
git commit -m "feat(phase2): add dispatch, round registry, resume, phase2 results writer"
```

---

## Task 4: phase2_state.py — CLI 封装

**Files:**
- Modify: `scripts/phase2_state.py`
- Modify: `tests/test_phase2_state.py`

CLI 是薄壳,供 SKILL.md 调用。子命令:`init` / `resume` / `gate` / `dispatch` / `backbone-get`。其余可变状态(promote/round/diagnosed/tuning)由 SKILL 通过这些函数的脚本入口或后续封装调用;本任务覆盖循环编排最常用的读侧命令,返回 JSON 便于 agent 解析。

- [ ] **Step 1: 追加 CLI 测试(subprocess 调用)**

```python
import subprocess
import sys


def test_cli_init_and_resume(toy_run):
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "init",
                        "--run-dir", str(toy_run), "--tag", "toy-tag"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert (toy_run / "phase2" / "state.json").exists()
    r2 = subprocess.run([sys.executable, "scripts/phase2_state.py", "resume",
                         "--run-dir", str(toy_run)], capture_output=True, text=True)
    assert r2.returncode == 0
    assert json.loads(r2.stdout)["action"] == "diagnose"


def test_cli_gate(toy_run):
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "gate",
                        "--candidate", "0.42", "--backbone", "0.40"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert json.loads(r.stdout)["promote"] is True
    r2 = subprocess.run([sys.executable, "scripts/phase2_state.py", "gate",
                         "--candidate", "0.41", "--backbone", "0.40"],
                        capture_output=True, text=True)
    assert json.loads(r2.stdout)["promote"] is False


def test_cli_dispatch(toy_run):
    exps = json.dumps([{"slot": "a", "needs_gpus": 1, "is_training": True},
                       {"slot": "b", "needs_gpus": 1, "is_training": False}])
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "dispatch",
                        "--experiments", exps, "--free-gpus", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["assigned"] == {"b": [0]}
    assert out["queued"] == ["a"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_phase2_state.py -k cli -q`
Expected: FAIL(无 `__main__` 或子命令未实现,returncode != 0)

- [ ] **Step 3: 追加 CLI 到 phase2_state.py 末尾**

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase-2 状态机 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    si = sub.add_parser("init", help="从 results.tsv 初始化主干")
    si.add_argument("--run-dir", type=Path, required=True)
    si.add_argument("--tag", required=True)

    sr = sub.add_parser("resume", help="打印下一步动作 JSON")
    sr.add_argument("--run-dir", type=Path, required=True)

    sg = sub.add_parser("gate", help="晋升门判定,打印 {promote: bool}")
    sg.add_argument("--candidate", type=float, required=True)
    sg.add_argument("--backbone", type=float, required=True)

    sd = sub.add_parser("dispatch", help="按空闲卡派发,打印分配 JSON")
    sd.add_argument("--experiments", required=True, help="JSON 数组字符串")
    sd.add_argument("--free-gpus", required=True, help="逗号分隔 GPU id,如 0,1,2")

    sb = sub.add_parser("backbone-get", help="打印当前主干 JSON")
    sb.add_argument("--run-dir", type=Path, required=True)

    args = ap.parse_args(argv)

    if args.cmd == "init":
        state = init_state(args.run_dir, args.tag)
        print(json.dumps(state["backbone"], ensure_ascii=False))
        return 0
    if args.cmd == "resume":
        state = load_state(args.run_dir) or {}
        print(json.dumps(next_action(state), ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        print(json.dumps({"promote": is_significant(args.candidate, args.backbone)}))
        return 0
    if args.cmd == "dispatch":
        exps = json.loads(args.experiments)
        gpus = [int(x) for x in args.free_gpus.split(",") if x != ""]
        print(json.dumps(plan_dispatch(exps, gpus), ensure_ascii=False))
        return 0
    if args.cmd == "backbone-get":
        state = load_state(args.run_dir) or {}
        print(json.dumps(state.get("backbone", {}), ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_phase2_state.py -q`
Expected: PASS(全部 18 个测试绿)

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2_state.py tests/test_phase2_state.py
git commit -m "feat(phase2): add phase2_state CLI (init/resume/gate/dispatch)"
```

---

## Task 5: diagnose.py — 子集视图短板诊断

**Files:**
- Create: `scripts/diagnose.py`
- Test: `tests/test_diagnose.py`

把冻结 `evaluate.py` 当黑盒,在 dataset 子集视图上反复调用,得全集/逐样本/分组分解。绝不重实现指标。

- [ ] **Step 1: 写失败测试**

```python
import json
import sys
from pathlib import Path

import pytest

from scripts import diagnose
from tests.helpers import write_predictions


def test_make_view_subsets_metadata_and_symlinks(toy_run, tmp_path):
    view = tmp_path / "view"
    diagnose._make_view(toy_run / "dataset", ["s1", "s3"], view)
    meta = json.loads((view / "metadata.json").read_text())
    assert [s["image_id"] for s in meta["samples"]] == ["s1", "s3"]
    assert (view / "s1").exists() and (view / "s3").exists()
    assert not (view / "s2").exists()


def test_detect_group_fields(toy_run):
    samples = json.loads((toy_run / "dataset" / "metadata.json").read_text())["samples"]
    assert diagnose._detect_group_fields(samples) == ["group"]


def test_detect_group_fields_degrades_when_all_unique(tmp_path):
    samples = [{"image_id": f"s{i}", "uid": f"u{i}"} for i in range(4)]
    # uid 全不同(4 distinct = n),超过 0.5*n 阈值 → 不分组
    assert diagnose._detect_group_fields(samples) == []


def test_diagnose_full_per_sample_worst_and_groups(toy_run, tmp_path):
    preds = toy_run / "predictions" / "predictions.jsonl"
    write_predictions(preds, {"s1": 0.9, "s2": 0.1, "s3": 0.5})  # group A=[.9,.1], B=[.5]
    work = tmp_path / "diagwork"
    out = diagnose.diagnose(preds, toy_run / "dataset", toy_run / "evaluate.py",
                            worst_k=2, work_dir=work)
    # full = mean(.9,.1,.5) = .5
    assert out["full"]["primary_metric"] == pytest.approx(0.5)
    # per-sample 升序,worst_k=2 → s2(.1), s3(.5)
    assert [w["id"] for w in out["worst_k"]] == ["s2", "s3"]
    # 分组:A=mean(.9,.1)=.5, B=.5
    assert out["groups"]["group"]["A"] == pytest.approx(0.5)
    assert out["groups"]["group"]["B"] == pytest.approx(0.5)


def test_diagnose_cli_writes_json_and_md(toy_run, tmp_path):
    preds = toy_run / "predictions" / "predictions.jsonl"
    write_predictions(preds, {"s1": 0.9, "s2": 0.1, "s3": 0.5})
    out_json = tmp_path / "diag.json"
    out_md = tmp_path / "diag.md"
    import subprocess
    r = subprocess.run([sys.executable, "scripts/diagnose.py",
                        "--predictions", str(preds),
                        "--dataset", str(toy_run / "dataset"),
                        "--evaluate-py", str(toy_run / "evaluate.py"),
                        "--out-json", str(out_json), "--out-md", str(out_md),
                        "--worst-k", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out_json.read_text())["full"]["primary_metric"] == pytest.approx(0.5)
    assert "worst" in out_md.read_text().lower()
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_diagnose.py -q`
Expected: FAIL,`ModuleNotFoundError: scripts.diagnose`

- [ ] **Step 3: 实现 diagnose.py**

```python
"""短板诊断 (只读,不改冻结 evaluate.py)。把 evaluate.py 当黑盒,
在 dataset 子集视图上反复调用,定位主干短板。绝不重实现指标。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_evaluate(evaluate_py: Path, predictions: Path, dataset_view: Path,
                  out_json: Path) -> dict:
    subprocess.run([sys.executable, str(evaluate_py),
                    "--predictions", str(predictions),
                    "--dataset", str(dataset_view),
                    "--out", str(out_json)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return json.loads(out_json.read_text())


def _make_view(dataset_dir: Path, sample_ids: list[str], view_dir: Path) -> None:
    """建子集视图:子集化 metadata.json + symlink 样本目录。原 dataset 不动。"""
    view_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    keep = set(sample_ids)
    sub = {k: v for k, v in meta.items() if k != "samples"}
    sub["samples"] = [s for s in meta["samples"] if s["image_id"] in keep]
    (view_dir / "metadata.json").write_text(json.dumps(sub, ensure_ascii=False))
    for sid in sample_ids:
        link = view_dir / sid
        if not link.exists():
            os.symlink((dataset_dir / sid).resolve(), link)


def _detect_group_fields(samples: list[dict], max_card_ratio: float = 0.5) -> list[str]:
    """自动探测可分组的分类字段:所有样本都有、值为 str/int/bool、2≤基数≤0.5*n。"""
    n = len(samples)
    if n == 0:
        return []
    keys = set().union(*[set(s.keys()) for s in samples]) - {"image_id"}
    fields = []
    for k in sorted(keys):
        vals = [s.get(k) for s in samples if isinstance(s.get(k), (str, int, bool))]
        if len(vals) != n:
            continue
        distinct = set(vals)
        if 2 <= len(distinct) <= max(2, int(n * max_card_ratio)):
            fields.append(k)
    return fields


def diagnose(predictions: Path, dataset_dir: Path, evaluate_py: Path,
             worst_k: int = 10, work_dir: Path | None = None) -> dict:
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    samples = meta["samples"]
    ids = [s["image_id"] for s in samples]
    tmp = Path(work_dir or tempfile.mkdtemp(prefix="diag_"))
    tmp.mkdir(parents=True, exist_ok=True)

    full = _run_evaluate(evaluate_py, predictions, dataset_dir, tmp / "full.json")

    per_sample = []
    for sid in ids:
        v = tmp / f"view_{sid}"
        _make_view(dataset_dir, [sid], v)
        r = _run_evaluate(evaluate_py, predictions, v, tmp / f"s_{sid}.json")
        per_sample.append({"id": sid, "primary": r["primary_metric"]})
    per_sample.sort(key=lambda x: x["primary"])
    worst = per_sample[:worst_k]

    groups: dict[str, dict] = {}
    for field in _detect_group_fields(samples):
        by_val: dict[str, list[str]] = {}
        for s in samples:
            by_val.setdefault(str(s[field]), []).append(s["image_id"])
        gmap = {}
        for val, sids in by_val.items():
            v = tmp / f"view_{field}_{val}"
            _make_view(dataset_dir, sids, v)
            r = _run_evaluate(evaluate_py, predictions, v, tmp / f"g_{field}_{val}.json")
            gmap[val] = r["primary_metric"]
        groups[field] = gmap

    return {"full": full, "per_sample": per_sample, "worst_k": worst,
            "groups": groups, "secondary_summary": full.get("metrics", {})}


def _to_md(diag: dict) -> str:
    lines = ["# 短板诊断", "",
             f"- primary_metric (全集): {diag['full']['primary_metric']:.6f}",
             "- 副指标: " + json.dumps(diag["secondary_summary"], ensure_ascii=False),
             "", "## 最差样本 (worst-K)"]
    for w in diag["worst_k"]:
        lines.append(f"- {w['id']}: {w['primary']:.6f}")
    if diag["groups"]:
        lines.append("\n## 分组得分")
        for field, gmap in diag["groups"].items():
            lines.append(f"### {field}")
            for val, sc in sorted(gmap.items(), key=lambda kv: kv[1]):
                lines.append(f"- {val}: {sc:.6f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="短板诊断 (黑盒调用冻结 evaluate.py)")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--evaluate-py", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--worst-k", type=int, default=10)
    args = ap.parse_args(argv)
    diag = diagnose(args.predictions, args.dataset, args.evaluate_py, args.worst_k)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2))
    args.out_md.write_text(_to_md(diag))
    print(f"diagnosed: primary={diag['full']['primary_metric']:.6f}, "
          f"worst={len(diag['worst_k'])}, groups={list(diag['groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_diagnose.py -q`
Expected: PASS(5 个测试绿)

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose.py tests/test_diagnose.py
git commit -m "feat(phase2): add diagnose.py — subset-view weakness analysis"
```

---

## Task 6: directions_schema.py — 方向 JSON 校验

**Files:**
- Create: `scripts/directions_schema.py`
- Test: `tests/test_directions_schema.py`

- [ ] **Step 1: 写失败测试**

```python
import json
import sys
import subprocess
from pathlib import Path

from scripts import directions_schema as dsx


def _valid():
    return [
        {"slot": "a", "title": "Method A", "source_urls": ["http://arxiv.org/abs/1"],
         "idea": "do x", "tier": "config", "needs_training": False},
        {"slot": "b", "title": "Method B", "source_urls": [],
         "idea": "do y", "tier": "train", "needs_training": True},
    ]


def test_validate_accepts_valid():
    assert dsx.validate(_valid()) == []


def test_validate_rejects_bad_tier():
    d = _valid()
    d[0]["tier"] = "magic"
    errs = dsx.validate(d)
    assert any("tier" in e for e in errs)


def test_validate_rejects_missing_field():
    d = _valid()
    del d[1]["idea"]
    errs = dsx.validate(d)
    assert any("idea" in e for e in errs)


def test_validate_rejects_duplicate_slots():
    d = _valid()
    d[1]["slot"] = "a"
    errs = dsx.validate(d)
    assert any("duplicate" in e for e in errs)


def test_validate_rejects_empty():
    assert dsx.validate([]) != []


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid()))
    r = subprocess.run([sys.executable, "scripts/directions_schema.py",
                        "--file", str(good)], capture_output=True, text=True)
    assert r.returncode == 0
    bad = tmp_path / "bad.json"
    d = _valid(); d[0]["tier"] = "magic"
    bad.write_text(json.dumps(d))
    r2 = subprocess.run([sys.executable, "scripts/directions_schema.py",
                         "--file", str(bad)], capture_output=True, text=True)
    assert r2.returncode == 1
    assert "tier" in r2.stdout + r2.stderr
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_directions_schema.py -q`
Expected: FAIL,`ModuleNotFoundError: scripts.directions_schema`

- [ ] **Step 3: 实现 directions_schema.py**

```python
"""directions.json schema 校验 (薄)。搜索本身由 agent 完成,本脚本只规范结构。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = {"slot": str, "title": str, "source_urls": list, "idea": str,
            "tier": str, "needs_training": bool}
TIERS = {"config", "post-process", "pipeline", "train", "infer-tune"}


def validate(directions) -> list[str]:
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
        if d.get("tier") not in TIERS:
            errs.append(f"[{i}] tier must be one of {sorted(TIERS)}")
        if not all(isinstance(u, str) for u in d.get("source_urls", [])):
            errs.append(f"[{i}] source_urls must be list[str]")
        slots.append(d.get("slot"))
    if len(set(slots)) != len(slots):
        errs.append(f"duplicate slots: {slots}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="directions.json 校验")
    ap.add_argument("--file", type=Path, required=True)
    args = ap.parse_args(argv)
    errs = validate(json.loads(args.file.read_text()))
    if errs:
        for e in errs:
            print(e)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_directions_schema.py -q`
Expected: PASS(6 个测试绿)

- [ ] **Step 5: Commit**

```bash
git add scripts/directions_schema.py tests/test_directions_schema.py
git commit -m "feat(phase2): add directions_schema validator"
```

---

## Task 7: train_launch.py — 通用训练壳

**Files:**
- Create: `scripts/train_launch.py`
- Test: `tests/test_train_launch.py`

不是 trainer;真正训练代码由 agent 写成 `train.py` 放进 exp_dir。本壳标准化四件易错事:数据出处记录、多卡启动命令拼装、checkpoint 续选、预算执行(超时 kill)。

- [ ] **Step 1: 写失败测试**

```python
import json
import time
from pathlib import Path

from scripts import train_launch as tl


def test_latest_checkpoint_picks_newest(tmp_path):
    ck = tmp_path / "ckpts"
    ck.mkdir()
    a = ck / "step100.pt"; a.write_text("x")
    time.sleep(0.01)
    b = ck / "step200.pt"; b.write_text("y")
    assert tl.latest_checkpoint(ck) == b


def test_latest_checkpoint_none_when_empty(tmp_path):
    assert tl.latest_checkpoint(tmp_path / "nope") is None


def test_record_provenance(tmp_path):
    tl.record_provenance(tmp_path, "CrelloTrain", "modelscope", "caches/modelscope/x")
    p = json.loads((tmp_path / "dataset_provenance.json").read_text())
    assert p["dataset"] == "CrelloTrain"
    assert p["source"] == "modelscope"


def test_build_launch_cmd_multi_gpu_and_resume(tmp_path):
    train_py = tmp_path / "train.py"
    train_py.write_text("")
    cmd = tl.build_launch_cmd(train_py, gpus=[0, 1, 2], extra_args=["--lr", "1e-4"],
                              resume_from=tmp_path / "ckpts" / "step200.pt")
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=3" in cmd
    assert "--lr" in cmd and "1e-4" in cmd
    assert "--resume" in cmd


def test_build_launch_cmd_no_resume(tmp_path):
    train_py = tmp_path / "train.py"; train_py.write_text("")
    cmd = tl.build_launch_cmd(train_py, gpus=[0], extra_args=[], resume_from=None)
    assert "--resume" not in cmd


def test_run_with_budget_times_out(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["sleep", "5"], log, gpus=[0], budget_seconds=1)
    assert status == "timeout"


def test_run_with_budget_done(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["true"], log, gpus=[0], budget_seconds=5)
    assert status == "done"
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_train_launch.py -q`
Expected: FAIL,`ModuleNotFoundError: scripts.train_launch`

- [ ] **Step 3: 实现 train_launch.py**

```python
"""通用训练壳:数据出处记录 / 多卡启动命令拼装 / checkpoint 续选 / 预算执行。
不是 trainer;真正训练代码由 agent 写成 train.py 放进 exp_dir。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

CKPT_SUFFIXES = {".pt", ".bin", ".safetensors"}


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """选 ckpt_dir 下最新 (按 mtime) 的权重文件或子目录,无则 None。"""
    if not ckpt_dir.exists():
        return None
    cands = [p for p in ckpt_dir.iterdir()
             if p.is_dir() or p.suffix in CKPT_SUFFIXES]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def record_provenance(exp_dir: Path, dataset: str, source: str, local_path: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "dataset_provenance.json").write_text(json.dumps(
        {"dataset": dataset, "source": source, "local_path": local_path},
        ensure_ascii=False, indent=2))


def build_launch_cmd(train_py: Path, gpus: list[int], extra_args: list[str],
                     resume_from: Path | None) -> list[str]:
    cmd = ["torchrun", f"--nproc_per_node={len(gpus)}", str(train_py), *extra_args]
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
    return cmd


def run_with_budget(cmd: list[str], log_path: Path, gpus: list[int],
                    budget_seconds: int) -> str:
    """跑命令,输出重定向到 log,超预算 kill。返回 'done'|'timeout'|'error'。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=budget_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return "timeout"
    return "done" if rc == 0 else "error"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="通用训练壳")
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--train-py", type=Path, required=True)
    ap.add_argument("--gpus", required=True, help="逗号分隔,如 0,1,2")
    ap.add_argument("--budget-seconds", type=int, required=True)
    ap.add_argument("--resume", action="store_true", help="从最新 ckpt 续训")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="透传给 train.py 的参数,可重复")
    args = ap.parse_args(argv)
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    ckpt_dir = args.exp_dir / "ckpts"
    resume_from = latest_checkpoint(ckpt_dir) if args.resume else None
    cmd = build_launch_cmd(args.train_py, gpus, args.train_arg, resume_from)
    status = run_with_budget(cmd, args.exp_dir / "train.log", gpus, args.budget_seconds)
    print(json.dumps({"status": status, "resumed_from": str(resume_from) if resume_from else None}))
    return 0 if status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行,确认通过**

Run: `uv run pytest tests/test_train_launch.py -q`
Expected: PASS(7 个测试绿;`run_with_budget` 超时用例约 1 秒)

- [ ] **Step 5: Commit**

```bash
git add scripts/train_launch.py tests/test_train_launch.py
git commit -m "feat(phase2): add train_launch.py — provenance, launch, ckpt-resume, budget"
```

---

## Task 8: 端到端 dry-run 契约测试

**Files:**
- Create: `tests/test_contract_phase2.py`

模拟一整轮的文件流转,锁住跨脚本接口:init → diagnose → open_round → 模拟 3 个 exp 的 metrics.json → record_slot → promote_check(gate)→ promote_backbone → 追加 results.tsv。Docker/推理/训练用产物文件直接 stub(只验证记账与 schema)。

- [ ] **Step 1: 写契约测试**

```python
import json
import csv
from pathlib import Path

import pytest

from scripts import phase2_state as ps
from scripts import diagnose
from tests.helpers import write_predictions


def test_full_round_file_flow(toy_run):
    # 0. init 主干
    state = ps.init_state(toy_run, "toy-tag")
    assert ps.next_action(state)["action"] == "diagnose"

    # 1. 诊断当前主干(用一份 backbone predictions)
    bb_preds = toy_run / "phase2" / "backbone_preds.jsonl"
    write_predictions(bb_preds, {"s1": 0.5, "s2": 0.2, "s3": 0.41})
    diag = diagnose.diagnose(bb_preds, toy_run / "dataset", toy_run / "evaluate.py",
                             worst_k=2, work_dir=toy_run / "phase2" / "diagwork")
    assert diag["full"]["primary_metric"] == pytest.approx(0.37, abs=1e-6)
    ps.mark_diagnosed(toy_run, state)
    ps.set_inference_tuning(toy_run, state, "explored")
    assert ps.next_action(state)["action"] == "search"

    # 2. 开一轮 3 方向
    dirs = [{"slot": "a", "tier": "config", "title": "A", "source_urls": ["u1"]},
            {"slot": "b", "tier": "pipeline", "title": "B", "source_urls": ["u2"]},
            {"slot": "c", "tier": "train", "title": "C", "source_urls": ["u3"]}]
    state = ps.open_round(toy_run, state, dirs)
    rid = state["rounds"][-1]["id"]

    # 3. 模拟 3 个 exp 推理+评分(写 predictions → 真跑玩具 evaluate via diagnose.full 路径)
    #    这里直接用 evaluate.py 算每个 exp 的 metrics.json,模拟 compute_metrics 产物。
    import subprocess, sys
    exp_scores = {"a": {"s1": 0.45, "s2": 0.45, "s3": 0.45},   # mean .45 (+12.5%)
                  "b": {"s1": 0.40, "s2": 0.40, "s3": 0.40},   # mean .40 (持平)
                  "c": {"s1": 0.10, "s2": 0.10, "s3": 0.10}}   # mean .10
    for slot, scores in exp_scores.items():
        exp_dir = toy_run / "phase2" / "rounds" / rid / f"exp_{slot}"
        preds = exp_dir / "predictions.jsonl"
        write_predictions(preds, scores)
        metrics_json = exp_dir / "metrics.json"
        subprocess.run([sys.executable, str(toy_run / "evaluate.py"),
                        "--predictions", str(preds),
                        "--dataset", str(toy_run / "dataset"),
                        "--out", str(metrics_json)], check=True)
        m = json.loads(metrics_json.read_text())
        state = ps.record_slot(toy_run, state, rid, slot, m["primary_metric"], "done")

    assert state["rounds"][-1]["status"] == "scored"
    assert ps.next_action(state)["action"] == "promote_check"

    # 4. 晋升判定:最佳是 a=.45 vs 主干 .40 → +12.5% ≥ 5% → 晋升
    slots = state["rounds"][-1]["slots"]
    best = max(slots, key=lambda s: s["primary_metric"])
    assert best["slot"] == "a"
    assert ps.is_significant(best["primary_metric"], state["backbone"]["primary_metric"])
    base_ver = state["backbone"]["version_n"]  # 晋升前的主干版本(记录到 results 行)
    state = ps.promote_backbone(toy_run, state, f"{rid}:exp_a",
                                f"rounds/{rid}/exp_a", best["primary_metric"],
                                {"score_mean": best["primary_metric"]})
    ps.append_phase2_result(toy_run, rid, "exp_a", base_ver, best["primary_metric"],
                            best["delta_pct"], "keep", "promoted from config tweak")
    state = ps.close_round(toy_run, state, rid)

    # 5. 断言全链路 schema + 状态
    assert state["backbone"]["version_n"] == 1
    assert state["backbone"]["primary_metric"] == pytest.approx(0.45)
    assert state["inference_tuning"] == "pending"          # 新主干重置便宜调优
    assert ps.next_action(state)["action"] == "diagnose"   # 回到诊断,形成螺旋

    # results.tsv schema(phase-2 7 列)
    rows = list(csv.reader((toy_run / "phase2" / "results.tsv").open(), delimiter="\t"))
    assert rows[0] == ["round", "exp", "base_backbone_ver", "primary_metric",
                       "delta_pct", "status", "description"]
    assert rows[1][5] == "keep"
    assert rows[1][4] == "12.50"   # delta% 相对晋升前主干 0.40

    # state.json 落盘且字段齐全
    loaded = ps.load_state(toy_run)
    assert loaded["backbone"]["id"] == f"{rid}:exp_a"
    assert loaded["directions_tried"]  # 3 方向已登记
```

- [ ] **Step 2: 运行,确认通过**

Run: `uv run pytest tests/test_contract_phase2.py -q`
Expected: PASS(锁住 diag→directions→metrics→record→promote→results.tsv/state.json 全链路)

- [ ] **Step 3: Commit**

```bash
git add tests/test_contract_phase2.py
git commit -m "test(phase2): add end-to-end dry-run contract test for the optimize loop"
```

---

## Task 9: GPU 冒烟测试(默认 skip)

**Files:**
- Create: `tests/test_smoke_phase2_gpu.py`

仅在真实 GPU 服务器手动跑,验证 phase2_state 的派发输出能被真实 `gpu_select.py` 喂饱、且训练壳能在真容器里启动一个最小 train.py。标 `@pytest.mark.gpu`,默认 skip(沿用 phase-1 marker)。

- [ ] **Step 1: 写冒烟测试**

```python
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu


def test_gpu_select_feeds_dispatch():
    """真实选卡 → 喂给 plan_dispatch,断言派发不为空。"""
    from scripts import phase2_state as ps
    r = subprocess.run([sys.executable, "scripts/gpu_select.py",
                        "--count", "1", "--min-free-mib", "1000"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no free GPU")
    gpus = [int(x) for x in r.stdout.strip().split(",") if x != ""]
    plan = ps.plan_dispatch([{"slot": "a", "needs_gpus": 1, "is_training": False}], gpus)
    assert plan["assigned"].get("a")


def test_train_launch_runs_minimal_train_py(tmp_path):
    """真起一个最小 train.py(纯 CPU 即可),验证壳能拼命令并跑完。"""
    from scripts import train_launch as tl
    exp = tmp_path / "exp"
    exp.mkdir()
    train_py = exp / "train.py"
    train_py.write_text("import sys; print('trained'); sys.exit(0)\n")
    # 用 python 直跑(不经 torchrun)验证 run_with_budget;torchrun 路径在真环境另测
    status = tl.run_with_budget([sys.executable, str(train_py)],
                                exp / "train.log", gpus=[0], budget_seconds=30)
    assert status == "done"
    assert "trained" in (exp / "train.log").read_text()
```

- [ ] **Step 2: 验证默认被 deselect**

Run: `uv run pytest tests/test_smoke_phase2_gpu.py -q`
Expected: `2 deselected`(`pyproject.toml` 的 `addopts = "-m 'not gpu'"` 默认排除 gpu 标记的测试——是 deselect 而非 skip)。在 GPU 服务器手动 `uv run pytest -m gpu tests/test_smoke_phase2_gpu.py` 跑真验证。

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_phase2_gpu.py
git commit -m "test(phase2): add gpu-marked smoke tests (default skip)"
```

---

## Task 10: references/optimization-loop.md — 优化循环细则

**Files:**
- Create: `skills/autoexplore-phase2/references/optimization-loop.md`

从 phase-2 SKILL.md 引用,装详细循环细则(避免主文件过长)。

- [ ] **Step 1: 写 optimization-loop.md**

````markdown
# 优化循环细则(第二阶段)

承接 SKILL.md。所有路径相对 repo 根;命令均从 repo 根 `uv run`。
状态唯一真相源 = `runs/<tag>/phase2/state.json`,只经 `scripts/phase2_state.py` 读写。
进入任意阶段前先 `resume` 决定从哪步续:

```bash
uv run scripts/phase2_state.py resume --run-dir runs/<tag>
# → {"action": "init|diagnose|infer_tune|search|execute|promote_check"}
```

## 步骤 0:Setup(唯一人工关卡)
1. 续用第一阶段 `<tag>`;`git checkout -b optimize/<tag>`(运行时专用分支)。
2. `uv run scripts/phase2_state.py init --run-dir runs/<tag> --tag <tag>`
   → 主干 = `results.tsv` 中 `status=ready` 且 primary 最高者。
3. 校验 `dataset/` 与 `evaluate.py` 在位且自第一阶段冻结后未变。确认后进入自主循环。

## 步骤 1:短板诊断(每次主干变化都重跑)
对当前主干已产出的 `predictions.jsonl`(初始主干来自第一阶段 `models/<name>/predictions.jsonl`):
```bash
uv run scripts/diagnose.py \
  --predictions <主干 predictions.jsonl> \
  --dataset runs/<tag>/dataset --evaluate-py runs/<tag>/evaluate.py \
  --out-json runs/<tag>/phase2/diagnostics/diag_<backbone_id>.json \
  --out-md   runs/<tag>/phase2/diagnostics/diag_<backbone_id>.md --worst-k 10
```
读 `.md`,把短板小结(哪些分组/样本/副指标最差)记进同名 `.md` 末尾,作为搜方向输入。
诊断后调用 `mark_diagnosed`(经脚本函数或后续封装)推进 `last_diagnosed_version`。

## 步骤 2:推理管道调优闸门(便宜档,先于训练)
判断:仅调主干推理管道(参数/后处理,无新方法、不训练)是否可能改善上面的短板?
- 有空间 → 按"实验执行"跑几个 `tier=infer-tune` 实验(config/post-process 组合);
  任一过门(见下)→ 晋升主干 → 回步骤 1。
- 无空间或都没过 → `set_inference_tuning explored` → 进步骤 3。
纪律:便宜档永远先穷尽,再动昂贵搜索。

## 步骤 3:不终止搜索优化循环
LOOP(直到人工终止):

### a. 搜方向(agent 判断)
在 Arxiv / HF / paperswithcode 针对短板检索;通用排序:代码/权重/数据集可得性、与短板相关性、报告增益、tier 成本。选 3 个,写 `runs/<tag>/phase2/rounds/<rid>/directions.json`,搜索过程进 `search.log`。校验并去重:
```bash
uv run scripts/directions_schema.py --file runs/<tag>/phase2/rounds/<rid>/directions.json
# 对每个方向 directions_seen 跳过已试(经 phase2_state 函数);open_round 会登记去重
```

### b. 派发并执行 3 个实验(按空闲卡并发,训练型排队)
```bash
GPUS=$(uv run scripts/gpu_select.py --count 8 --min-free-mib 20000 | tr ',' '\n')
uv run scripts/phase2_state.py dispatch \
  --experiments '[{"slot":"a","needs_gpus":1,"is_training":false}, ...]' \
  --free-gpus "$(echo $GPUS | tr ' ' ',')"
# → {"assigned": {"a":[0],...}, "queued": ["c"]}  训练型卡紧时进 queued,等卡再跑
```
每个 `exp_<slot>/` 按 tier 实现并产出 `predictions.jsonl`:
- **复用主干 docker 镜像**;需额外依赖才写 `FROM <主干镜像>` 的小 Dockerfile + `docker_env.py build`。
- `config`:换主干推理参数 · `post-process`:在主干预测上加后处理 · `pipeline`:串接组件 ·
  `train`:`train_launch.py` 取数据→多卡训→ckpt→用新权重推理 · `infer-tune`:config/post-process 组合。
- 推理:`run_inference.py`(复用第一阶段,挂 caches 与 dataset,带 `--user`/`--runtime nvidia`)。
- 评分:`compute_metrics.py --evaluate-py runs/<tag>/evaluate.py --predictions <exp>/predictions.jsonl --dataset runs/<tag>/dataset --out <exp>/metrics.json`。
- 逐实验 `progress.py` 记 stage,重试上限 3;crash 记一行不阻塞兄弟。
全部产出后 `record_slot`(done/crash),凑齐则轮转 `scored`。

### c–d. 算分与晋升
```bash
uv run scripts/phase2_state.py gate --candidate <最佳 exp primary> --backbone <主干 primary>
# {"promote": true}  当且仅当相对 ≥ +5%
```
- 过门 → `promote_backbone`(version_n++、移指针、重置 inference_tuning=pending)→ `git add <exp> && git commit` 留痕 → `close_round` → 回步骤 1(螺旋)。
- 不过门 → 三者各记 `results.tsv` 一行(`discard`/`crash`)→ `close_round` → 回 a 选新方向。

### e. 永不停下
绝不问人是否继续。"没主意"时:重读论文、组合差一点的近似命中、试更激进改动——直到人工终止。

## 日志与重试纪律
- 容器/训练输出进 `run.log`/`train.log`,只失败时 `tail -n 50`,绝不全量进上下文。
- 逐实验重试上限 3;crash 不阻塞同轮兄弟,也不停整循环。
- keep/discard 由 state.json 主干指针管理,**不用 git reset** 抹实验目录(失败也是研究档案)。
````

- [ ] **Step 2: 校验 Markdown 可读 + 引用路径存在**

Run: `test -f skills/autoexplore-phase2/references/optimization-loop.md && grep -c '步骤' skills/autoexplore-phase2/references/optimization-loop.md`
Expected: 输出 ≥ 4(四个主步骤标题在)

- [ ] **Step 3: Commit**

```bash
git add skills/autoexplore-phase2/references/optimization-loop.md
git commit -m "docs(phase2): add optimization-loop reference for the phase-2 skill"
```

---

## Task 11: SKILL.md — 第二阶段编排入口

**Files:**
- Create: `skills/autoexplore-phase2/SKILL.md`

- [ ] **Step 1: 写 SKILL.md**

````markdown
---
name: autoexplore-phase2
description: Use after phase-1 produced a baseline — drives phase-2 iterative optimization: set the chosen model as the backbone, diagnose its test-set weaknesses, try cheap inference-pipeline tuning first, then run a never-ending search loop (Arxiv/HF/paperswithcode → 3 directions → train if needed → score → promote on relative +5%).
---

# autoexplore 第二阶段:模型效果迭代优化

输入:第一阶段在 `runs/<tag>/` 产出的冻结 `dataset/`、`evaluate.py`、`results.tsv` 与 ≥1 个 `status=ready` 模型。
产出:在同方向上不终止地优化"模型主干",每次有明显提升(相对 +5%)就晋升新主干,直到人工终止。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡;**复用第一阶段主干的 docker 镜像**;按空闲度选卡。

**这是通用 skill,不绑定任何具体任务/数据集/模型。** 只依赖第一阶段冻结的文件契约
(`metrics.json` 的 `{primary_metric, metrics{}}`、`evaluate.py` 的 CLI、`results.tsv` 列),不依赖任务语义。

## 流程(需求文档第二阶段)

唯一真相源 = `runs/<tag>/phase2/state.json`,只经 `scripts/phase2_state.py` 读写。任意时刻先 `resume` 决定续点:
```bash
uv run scripts/phase2_state.py resume --run-dir runs/<tag>
```

### 步骤 0:Setup(唯一人工关卡)
续用第一阶段 `<tag>`;`git checkout -b optimize/<tag>`;`phase2_state init` 设主干 = ready 最高分;
校验 `dataset/`+`evaluate.py` 仍冻结。确认后进入自主循环,之后不再逐步问人。

### 步骤 1:短板诊断(每次主干变化重跑)
`diagnose.py` 把冻结 `evaluate.py` 当黑盒在子集视图上跑 → `diagnostics/diag_<id>.{json,md}`;
读 `.md` 写短板小结(最差分组/样本/副指标)。

### 步骤 2:推理管道调优闸门(便宜档先行)
判断"仅调推理管道(参数/后处理,不训练)"是否有改善空间;有就先跑 `infer-tune` 实验,
过门即晋升回步骤 1;无空间则 `inference_tuning=explored` 进步骤 3。

### 步骤 3:不终止搜索循环(复用主干镜像)
LOOP:搜 Arxiv/HF/paperswithcode 选 3 方向(`directions.json`,`directions_schema.py` 校验、跨轮去重)→
`dispatch` 按空闲卡并发、训练型排队 → 各 `exp_*/` 按 tier 实现(需训练用 `train_launch.py`)→
`run_inference`+`compute_metrics` 评分 → `gate` 判相对 +5% → 过门 `promote_backbone`+commit 回步骤 1,
否则记 `discard/crash` 继续选新方向。**绝不停下问人**,直到人工终止。

完整细则见 [references/optimization-loop.md](references/optimization-loop.md)。

## 关键纪律
- `dataset/` 与 `evaluate.py` 全程**不可变**,保证主干各版本与实验可比。
- **便宜档先行**:推理管道调优闸门先于昂贵搜索;搜索内 config/post-process 优先于训练。
- **晋升只认主指标相对 +5%**,避免噪声/随机种子的微小波动误晋升致主干抖动。
- **keep/discard 用 state.json 主干指针,不用 git reset**:失败实验目录保留作研究档案。
- 容器/训练输出进 log,只失败时 `tail`;逐实验重试上限 3,crash 不阻塞同轮兄弟。
- 中断可恢复:入口 `resume` 读 state.json;已 scored slot 跳过、已晋升主干不回退。
- **容器纪律(继承第一阶段)**:每次 `docker run` 带 `--user $UID:$GID --runtime=nvidia`;
  caches 以 `:ro` 挂 `/cache/{modelscope,huggingface,torch}`,env 注入对应 `*_CACHE`/`*_HOME`。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/phase2_state.py {init,resume,gate,dispatch,backbone-get}` | 状态/晋升门/派发(确定性核心) |
| `scripts/diagnose.py` | 黑盒调用 evaluate.py 出短板分解 |
| `scripts/directions_schema.py --file <directions.json>` | 方向 schema 校验 |
| `scripts/train_launch.py` | 数据出处/多卡启动/ckpt 续/预算 |
| `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py` | 复用第一阶段 |
````

- [ ] **Step 2: 校验 frontmatter + 引用**

Run: `head -4 skills/autoexplore-phase2/SKILL.md && test -f skills/autoexplore-phase2/references/optimization-loop.md && echo OK`
Expected: 打印 frontmatter(name: autoexplore-phase2)+ `OK`

- [ ] **Step 3: Commit**

```bash
git add skills/autoexplore-phase2/SKILL.md
git commit -m "feat(phase2): add autoexplore-phase2 skill entry (SKILL.md)"
```

---

## Task 12: 全量回归 + 完成标准核对

**Files:**(无新增)

- [ ] **Step 1: 跑全部非 GPU 测试**

Run: `uv run pytest -q -m "not gpu"`
Expected: 全绿(phase2_state 18 + diagnose 5 + directions 6 + train_launch 7 + contract 1 = 37 个新测试,叠加 phase-1 测试一并通过)

- [ ] **Step 2: 核对完成标准**

逐条确认:
- `scripts/phase2_state.py` 受测核心齐备(gate / dispatch / 主干 / 轮次 / resume / CLI)。
- `diagnose.py` 黑盒诊断、不改 `evaluate.py`,降级路径有测试。
- `directions_schema.py` 校验 tier/必填/去重。
- `train_launch.py` 出处/命令/ckpt/预算超时 kill 有测试。
- 契约测试锁住 diag→directions→metrics→record→promote→results.tsv/state.json 全链路与"晋升后回诊断"螺旋。
- `skills/autoexplore-phase2/SKILL.md` + `references/optimization-loop.md` 完整,frontmatter 合法。
- 全程未触碰第一阶段冻结的 `dataset/`、`evaluate.py`;未引入新运行时依赖。

- [ ] **Step 3:(可选)在 GPU 服务器手动跑冒烟**

Run: `uv run pytest -m gpu -q`
Expected: 在有空闲卡时通过;无卡自动 skip。

---

## 完成标准

1. `uv run pytest -m "not gpu"` 全绿;新脚本均 TDD 覆盖。
2. 第二阶段为**独立 skill**(`skills/autoexplore-phase2/`),phase-1 根布局不动,共享 `scripts/`。
3. 状态机是唯一真相源,支持中断恢复;晋升门 = 相对 +5%;派发训练型排队。
4. 诊断只读、把冻结 `evaluate.py` 当黑盒,通用不绑任务语义。
5. SKILL.md + reference 完整描述"诊断 → 便宜调优闸门 → 不终止搜索循环 → 相对 +5% 晋升螺旋",并保留第一阶段容器/缓存/日志纪律。
6. 运行时循环在 `optimize/<tag>` 分支用直接 commit + state.json 指针管 keep/discard(不用 git reset)。
