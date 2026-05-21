"""真实 Docker 冒烟测试。默认 skip,在 GPU 服务器上用 `pytest -m gpu` 跑。"""
import pytest
from scripts.docker_env import check_runtime, build_run_command, run_in_container

pytestmark = pytest.mark.gpu


def test_runtime_available():
    check_runtime()  # 不抛异常即通过


def test_container_echo_writes_log(tmp_path):
    log = tmp_path / "run.log"
    rc = run_in_container(
        image="python:3.12-slim", gpus="",
        mounts=[], inner_cmd="echo hello-autoexplore", log_path=log,
    )
    assert rc == 0
    assert "hello-autoexplore" in log.read_text()


def test_gpu_visible_in_container(tmp_path):
    """验证 --gpus 透传:容器内能看到 nvidia-smi。"""
    log = tmp_path / "gpu.log"
    rc = run_in_container(
        image="nvidia/cuda:12.4.0-base-ubuntu22.04", gpus="0",
        mounts=[], inner_cmd="nvidia-smi -L", log_path=log,
    )
    assert rc == 0
    assert "GPU 0" in log.read_text()
