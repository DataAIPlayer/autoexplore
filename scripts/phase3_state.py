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
