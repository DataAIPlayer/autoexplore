"""在容器内对 dataset 跑推理,产出 predictions.jsonl(每行一条 JSON)。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接 `uv run scripts/run_inference.py` 时,scripts 不在 sys.path;补上 repo 根目录
# 让 `from scripts.docker_env import ...` 解析得到。pytest 模式下走 cwd 不受影响。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.docker_env import run_in_container

# run 目录挂载到容器内 /work;推理脚本须把输出写到这里
CONTAINER_WORK = "/work"
INNER_PREDICTIONS_PATH = "/work/predictions.jsonl"


def run_inference(
    image: str, gpus: str, run_dir: Path, infer_cmd: str, log_path: Path,
    *,
    dataset_dir: Path | None = None,
    extra_mounts: list[tuple[str, str]] | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    runtime: str | None = "nvidia",
) -> Path:
    """挂载 run_dir 到 /work,执行 infer_cmd;成功返回宿主侧 predictions 路径。

    infer_cmd 由 Claude 提供,约定把预测写到 INNER_PREDICTIONS_PATH。

    新增:
      dataset_dir: 显式挂到 /work/dataset (容器看不到 run_dir 之外的 dataset 时用)
      extra_mounts: 任意额外 host:container 挂载 (HF/MS/torch cache 等)
      env / user / runtime: 透传给 docker_env.run_in_container,
        默认 runtime='nvidia' 处理 docker 默认 runtime 非 nvidia 的主机
    """
    # Docker bind-mount 要求绝对路径,defensively resolve 用户传进来的相对路径。
    mounts: list[tuple[str, str]] = [(str(Path(run_dir).resolve()), CONTAINER_WORK)]
    if dataset_dir is not None:
        mounts.append((str(Path(dataset_dir).resolve()), f"{CONTAINER_WORK}/dataset"))
    if extra_mounts:
        # extra mounts: host 侧也 resolve;若带 `:ro` 等选项保持原样
        for host, container in extra_mounts:
            # 支持 "host_path:container_path:flags" 中 container_path 含冒号的写法
            host_abs = str(Path(host).resolve()) if host else host
            mounts.append((host_abs, container))
    rc = run_in_container(
        image=image,
        gpus=gpus,
        mounts=mounts,
        inner_cmd=f"{infer_cmd} && test -f {INNER_PREDICTIONS_PATH}",
        log_path=log_path,
        env=env,
        user=user,
        runtime=runtime,
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
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="若 dataset 在 run_dir 之外,显式挂到 /work/dataset")
    ap.add_argument("--extra-mount", action="append", default=[],
                    help="host:container[:ro],可重复 (HF/MS/torch cache 等)")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VAL,可重复")
    ap.add_argument("--user", default=None, help="UID:GID")
    ap.add_argument("--runtime", default="nvidia",
                    help="docker --runtime,默认 nvidia;设空字符串则不带")
    args = ap.parse_args(argv)
    extra = [tuple(m.split(":", 1)) for m in args.extra_mount]
    env_dict = dict(kv.split("=", 1) for kv in args.env) if args.env else None
    runtime = args.runtime or None
    try:
        preds = run_inference(
            args.tag, args.gpus, args.run_dir, args.infer_cmd, args.log,
            dataset_dir=args.dataset_dir,
            extra_mounts=extra,
            env=env_dict,
            user=args.user,
            runtime=runtime,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"predictions: {preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
