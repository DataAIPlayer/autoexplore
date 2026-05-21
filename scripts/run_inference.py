"""在容器内对 dataset 跑推理,产出 predictions.jsonl(每行一条 JSON)。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.docker_env import run_in_container

# run 目录挂载到容器内 /work;推理脚本须把输出写到这里
CONTAINER_WORK = "/work"
INNER_PREDICTIONS_PATH = "/work/predictions.jsonl"


def run_inference(
    image: str, gpus: str, run_dir: Path, infer_cmd: str, log_path: Path
) -> Path:
    """挂载 run_dir 到 /work,执行 infer_cmd;成功返回宿主侧 predictions 路径。

    infer_cmd 由 Claude 提供,约定把预测写到 INNER_PREDICTIONS_PATH。
    """
    rc = run_in_container(
        image=image,
        gpus=gpus,
        mounts=[(str(run_dir), CONTAINER_WORK)],
        inner_cmd=f"{infer_cmd} && test -f {INNER_PREDICTIONS_PATH}",
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(f"推理失败 (exit={rc}),详见 {log_path}")
    return run_dir / "predictions.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="容器内推理")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--infer-cmd", required=True)
    args = ap.parse_args(argv)
    try:
        preds = run_inference(
            args.tag, args.gpus, args.run_dir, args.infer_cmd, args.log
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"predictions: {preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
