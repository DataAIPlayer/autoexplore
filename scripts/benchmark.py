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
