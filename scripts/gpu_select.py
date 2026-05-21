"""按显卡空闲显存挑选 GPU,打印 CUDA_VISIBLE_DEVICES 供容器使用。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx, total, used = (p.strip() for p in line.split(","))
        gpus.append(GpuInfo(int(idx), int(total), int(used)))
    return gpus


def select_gpus(gpus: list[GpuInfo], count: int, min_free_mib: int = 0) -> list[int]:
    eligible = [g for g in gpus if g.free_mib >= min_free_mib]
    if not eligible:
        raise RuntimeError(
            f"无空闲 GPU:没有显卡满足空闲显存 >= {min_free_mib} MiB"
        )
    eligible.sort(key=lambda g: g.free_mib, reverse=True)
    return [g.index for g in eligible[:count]]


def query_gpus() -> list[GpuInfo]:
    try:
        out = subprocess.run(
            NVIDIA_SMI_QUERY, capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError as e:
        raise RuntimeError("找不到 nvidia-smi:本机无 NVIDIA runtime") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 执行失败: {e.stderr}") from e
    return parse_nvidia_smi(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按空闲度选 GPU")
    ap.add_argument("--count", type=int, default=1, help="需要的 GPU 数量")
    ap.add_argument("--min-free-mib", type=int, default=0, help="每卡最小空闲显存(MiB)")
    args = ap.parse_args(argv)
    try:
        chosen = select_gpus(query_gpus(), args.count, args.min_free_mib)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(",".join(str(i) for i in chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
