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
    eligible = [c for c in candidates if c.get("passes_quality") and c.get("latency_ms") is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda c: c["latency_ms"])


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
    directions_add(run_dir, state, directions)  # 已含 save_state
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
    if status != "crash":
        target["speedup_pct"] = speedup_ratio(base_lat, latency_ms) * 100.0
        target["quality_loss_pct"] = quality_loss_ratio(baseline_q, quality) * 100.0
    else:
        target["speedup_pct"] = None
        target["quality_loss_pct"] = None
    target["status"] = status
    rnd["status"] = ("scored"
                     if all(s["status"] in ("done", "crash") for s in rnd["slots"])
                     else "running")
    save_state(run_dir, state)
    return state


def close_round(run_dir: Path, state: dict, round_id: str) -> dict:
    _find_round(state, round_id)["status"] = "done"
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
        if state["baseline"].get("latency_ms") is None:
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
