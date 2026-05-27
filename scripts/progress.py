"""复现进度持久化(progress.json)与结果汇总(results.tsv)。"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from pathlib import Path

PROGRESS_FILE = "progress.json"
RESULTS_FILE = "results.tsv"
TERMINAL_STAGES = {"ready", "crash"}
RESULTS_HEADER = ["model", "primary_metric", "memory_gb", "status", "description"]


@dataclass
class Progress:
    model: str
    stage: str  # A/B/C/ready/crash
    retry_count: int = 0
    gpus: str = ""
    image_tag: str = ""
    last_error: str = ""
    updated_at: str = ""
    # 推理配置快照 (steps/resolution/layers/shards 等),便于"是配置变了还是网络抖了"的复盘
    infer_config: dict[str, str] = field(default_factory=dict)


def save_progress(model_dir: Path, progress: Progress) -> None:
    progress = replace(progress, updated_at=datetime.now(timezone.utc).isoformat())
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / PROGRESS_FILE).write_text(
        json.dumps(asdict(progress), ensure_ascii=False, indent=2)
    )


def load_progress(model_dir: Path) -> Progress | None:
    path = model_dir / PROGRESS_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # 向后兼容:旧 progress.json 没有 infer_config 字段
    data.setdefault("infer_config", {})
    return Progress(**data)


def is_done(model_dir: Path) -> bool:
    p = load_progress(model_dir)
    return p is not None and p.stage in TERMINAL_STAGES


def append_result(
    run_dir: Path, model: str, primary_metric: float,
    memory_gb: float, status: str, description: str,
) -> None:
    path = run_dir / RESULTS_FILE
    is_new = not path.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if is_new:
            w.writerow(RESULTS_HEADER)
        w.writerow([
            model, f"{primary_metric:.6f}", f"{memory_gb:.1f}",
            status, description,
        ])


def read_results(run_dir: Path) -> list[list[str]]:
    path = run_dir / RESULTS_FILE
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="进度与结果工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("set", help="写入 progress.json")
    s.add_argument("--model-dir", type=Path, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--stage", required=True)
    s.add_argument("--retry-count", type=int, default=0)
    s.add_argument("--gpus", default="")
    s.add_argument("--image-tag", default="")
    s.add_argument("--last-error", default="")
    s.add_argument("--infer-config", action="append", default=[],
                   help="KEY=VAL,可重复 (steps/resolution/layers/shards 等)")

    g = sub.add_parser("done", help="查询模型是否处于终态,done 退出 0 否则 1")
    g.add_argument("--model-dir", type=Path, required=True)

    r = sub.add_parser("result", help="追加一行 results.tsv")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--primary-metric", type=float, required=True)
    r.add_argument("--memory-gb", type=float, required=True)
    r.add_argument("--status", required=True)
    r.add_argument("--description", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "set":
        cfg = dict(kv.split("=", 1) for kv in args.infer_config) if args.infer_config else {}
        save_progress(args.model_dir, Progress(
            model=args.model, stage=args.stage, retry_count=args.retry_count,
            gpus=args.gpus, image_tag=args.image_tag, last_error=args.last_error,
            infer_config=cfg,
        ))
        return 0
    if args.cmd == "done":
        return 0 if is_done(args.model_dir) else 1
    if args.cmd == "result":
        append_result(
            args.run_dir, args.model, args.primary_metric,
            args.memory_gb, args.status, args.description,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
