"""通用训练壳:数据出处记录 / 多卡启动命令拼装 / checkpoint 续选 / 预算执行。
不是 trainer;真正训练代码由 agent 写成 train.py 放进 exp_dir。"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

CKPT_SUFFIXES = {".pt", ".bin", ".safetensors"}


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """选 ckpt_dir 下最新 (按 mtime) 的权重文件或子目录,无则 None。"""
    if not ckpt_dir.exists():
        return None
    cands = [p for p in ckpt_dir.iterdir()
             if p.is_dir() or p.suffix in CKPT_SUFFIXES]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def record_provenance(exp_dir: Path, dataset: str, source: str, local_path: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "dataset_provenance.json").write_text(json.dumps(
        {"dataset": dataset, "source": source, "local_path": local_path},
        ensure_ascii=False, indent=2))


def build_launch_cmd(train_py: Path, gpus: list[int], extra_args: list[str],
                     resume_from: Path | None) -> list[str]:
    cmd = ["torchrun", f"--nproc_per_node={len(gpus)}", str(train_py), *extra_args]
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
    return cmd


def _kill_process_group(proc: subprocess.Popen) -> None:
    """kill 整个进程组。torchrun 会派生 worker rank,只 kill launcher 会留孤儿占着 GPU,
    阻塞后续实验;start_new_session 让 proc 自成进程组,这里整组 SIGKILL。"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def run_with_budget(cmd: list[str], log_path: Path, gpus: list[int],
                    budget_seconds: int) -> str:
    """跑命令,输出重定向到 log,超预算 kill 整个进程组。返回 'done'|'timeout'|'error'。
    启动器不存在(如未装 torchrun)记 'error' 而非让编排器崩溃。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    with open(log_path, "w") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
        except FileNotFoundError as e:
            log.write(f"launcher not found: {e}\n")
            return "error"
        try:
            rc = proc.wait(timeout=budget_seconds)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            return "timeout"
    return "done" if rc == 0 else "error"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="通用训练壳")
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--train-py", type=Path, required=True)
    ap.add_argument("--gpus", required=True, help="逗号分隔,如 0,1,2")
    ap.add_argument("--budget-seconds", type=int, required=True)
    ap.add_argument("--resume", action="store_true", help="从最新 ckpt 续训")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="透传给 train.py 的参数,可重复")
    args = ap.parse_args(argv)
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    ckpt_dir = args.exp_dir / "ckpts"
    resume_from = latest_checkpoint(ckpt_dir) if args.resume else None
    cmd = build_launch_cmd(args.train_py, gpus, args.train_arg, resume_from)
    status = run_with_budget(cmd, args.exp_dir / "train.log", gpus, args.budget_seconds)
    print(json.dumps({"status": status, "resumed_from": str(resume_from) if resume_from else None}))
    return 0 if status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
