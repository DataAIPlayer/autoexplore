"""Multi-process (torchrun) wrapper around the FROZEN benchmark protocol for 3c.

Reuses scripts/benchmark.py's measure()/select_records() verbatim — the timing
protocol (fixed subset, warmup, iters, batch=1 single-record latency) is identical
to single-card. The ONLY differences: (1) a torch.distributed group is initialized
so the adapter can shard the DiT across ranks (TP/SP), (2) every rank runs the same
records in lockstep (collectives keep them in step; same seed => identical output),
(3) only rank 0 writes speed.json / predictions.jsonl.

Launch: torchrun --nproc_per_node=N benchmark_dist.py --adapter ... --bench-config ...
The adapter's load_model() reads RANK/WORLD_SIZE from the env / the initialized group.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark  # frozen single-card protocol (measure, load_adapter, select_records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Distributed (TP/SP) bench — frozen protocol, rank0 writes")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--bench-config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    bench_config = json.loads(Path(args.bench_config).read_text())
    mod = benchmark.load_adapter(Path(args.adapter))
    # measure() runs warmup + timed loop identically on every rank; the adapter's
    # sharded forward all-reduces, so ranks advance in lockstep. seed is fixed in
    # bench_config => identical predictions on every rank.
    speed, preds = benchmark.measure(mod, Path(args.dataset), bench_config)

    if rank == 0:
        speed["gpu_count"] = world
        Path(args.out).write_text(json.dumps(speed, ensure_ascii=False, indent=2))
        with Path(args.predictions).open("w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(json.dumps({"latency_mean_ms": speed["latency_mean_ms"],
                          "p50_ms": speed["p50_ms"], "world_size": world}))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
