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
    """候选是否相对当前主干显著提升 (>= backbone*(1+rel))。
    用微小 epsilon 消除浮点边界误差。"""
    if backbone <= 0.0:
        return candidate > 0.0
    return candidate >= backbone * (1.0 + rel) - 1e-9


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
    """status ∈ {done, crash}。完成的 slot 算 delta% vs 当前主干。全 slot 终态则轮转 scored。
    slot 不存在于该轮时抛 KeyError(对齐 _find_round,fail loud 而非静默 no-op)。"""
    rnd = _find_round(state, round_id)
    bb = state["backbone"]["primary_metric"]
    target = next((s for s in rnd["slots"] if s["slot"] == slot), None)
    if target is None:
        raise KeyError(f"slot {slot!r} not in round {round_id!r}")
    target["primary_metric"] = primary_metric
    target["delta_pct"] = ((primary_metric - bb) / bb * 100.0) if bb > 0 else None
    target["status"] = status
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
