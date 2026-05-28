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
