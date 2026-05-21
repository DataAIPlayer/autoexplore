"""Docker 环境检查、镜像构建复用、容器内执行命令。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class DockerError(RuntimeError):
    """Docker 不可用或操作失败。"""


def check_runtime() -> None:
    """验证 docker 可用且支持 GPU,缺失则 fail fast。"""
    try:
        info = subprocess.run(
            ["docker", "info"], capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise DockerError("找不到 docker:本机未安装或不在 PATH") from e
    if info.returncode != 0:
        raise DockerError(f"docker 不可用: {info.stderr.strip()}")


def image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def build_image(tag: str, dockerfile_dir: Path) -> bool:
    """镜像已存在则跳过,返回 False;否则构建并返回 True。"""
    if image_exists(tag):
        return False
    proc = subprocess.run(
        ["docker", "build", "-t", tag, str(dockerfile_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise DockerError(f"镜像构建失败: {proc.stderr.strip()}")
    return True


def build_run_command(
    image: str, gpus: str, mounts: list[tuple[str, str]], inner_cmd: str
) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", f'"device={gpus}"']
    for host, container in mounts:
        cmd += ["-v", f"{host}:{container}"]
    cmd += [image, "sh", "-c", inner_cmd]
    return cmd


def run_in_container(
    image: str, gpus: str, mounts: list[tuple[str, str]],
    inner_cmd: str, log_path: Path,
) -> int:
    """跑容器,stdout+stderr 重定向到 log_path,返回退出码。"""
    cmd = build_run_command(image, gpus, mounts, inner_cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Docker 环境工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="检查 docker runtime")

    b = sub.add_parser("build", help="构建/复用镜像")
    b.add_argument("--tag", required=True)
    b.add_argument("--dockerfile-dir", type=Path, required=True)

    r = sub.add_parser("run", help="容器内执行命令")
    r.add_argument("--tag", required=True)
    r.add_argument("--gpus", default="")
    r.add_argument("--mount", action="append", default=[], help="host:container")
    r.add_argument("--log", type=Path, required=True)
    r.add_argument("--inner-cmd", required=True)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "check":
            check_runtime()
            print("docker runtime OK")
            return 0
        if args.cmd == "build":
            built = build_image(args.tag, args.dockerfile_dir)
            print("built" if built else "reused")
            return 0
        if args.cmd == "run":
            mounts = [tuple(m.split(":", 1)) for m in args.mount]
            rc = run_in_container(
                args.tag, args.gpus, mounts, args.inner_cmd, args.log
            )
            print(f"exit_code: {rc}")
            return rc
    except DockerError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
