"""调用 run 专属的 evaluate.py,校验并归一成 metrics.json。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class MetricsError(RuntimeError):
    """评测失败:evaluate.py 崩溃或输出 schema 不合法。"""


def compute_metrics(
    evaluate_py: Path, predictions: Path, dataset_dir: Path, out: Path
) -> dict:
    cmd = [
        sys.executable, str(evaluate_py),
        "--predictions", str(predictions),
        "--dataset", str(dataset_dir),
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MetricsError(f"evaluate.py 执行失败: {proc.stderr.strip()}")
    if not out.exists():
        raise MetricsError("evaluate.py 未写出 metrics 文件")
    data = json.loads(out.read_text())
    if "primary_metric" not in data:
        raise MetricsError("metrics.json 缺少 primary_metric 字段")
    if not isinstance(data["primary_metric"], (int, float)):
        raise MetricsError("primary_metric 必须是数字")
    data.setdefault("metrics", {})
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="计算评价指标")
    ap.add_argument("--evaluate-py", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = compute_metrics(
            args.evaluate_py, args.predictions, args.dataset, args.out
        )
    except MetricsError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"primary_metric: {result['primary_metric']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
