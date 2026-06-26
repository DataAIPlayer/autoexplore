"""directions.json schema 校验 (薄)。搜索本身由 agent 完成,本脚本只规范结构。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = {"slot": str, "title": str, "source_urls": list, "idea": str,
            "tier": str, "needs_training": bool}
PHASE2_TIERS = {"config", "post-process", "pipeline", "train", "infer-tune"}
PHASE3_TIERS = {"framework", "quantization", "kernel", "decoding", "compile", "parallel"}
TIERS = PHASE2_TIERS                       # 默认与历史行为一致
TIER_SETS = {"phase2": PHASE2_TIERS, "phase3": PHASE3_TIERS,
             "all": PHASE2_TIERS | PHASE3_TIERS}


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


if __name__ == "__main__":
    raise SystemExit(main())
