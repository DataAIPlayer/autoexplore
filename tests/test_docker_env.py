from unittest import mock
import subprocess
import pytest
from scripts.docker_env import (
    image_exists, build_image, run_in_container,
    check_runtime, DockerError, build_run_command,
)

def test_image_exists_true_when_inspect_succeeds():
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        assert image_exists("autoexplore/llava") is True
        m.assert_called_once()
        assert "image" in m.call_args.args[0]
        assert "inspect" in m.call_args.args[0]

def test_image_exists_false_when_inspect_fails():
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 1, "", "No such image")
        assert image_exists("missing") is False

def test_build_image_skips_when_exists(tmp_path):
    with mock.patch("scripts.docker_env.image_exists", return_value=True), \
         mock.patch("subprocess.run") as m:
        built = build_image("autoexplore/llava", tmp_path)
        assert built is False        # 未构建
        m.assert_not_called()

def test_build_image_builds_when_missing(tmp_path):
    with mock.patch("scripts.docker_env.image_exists", return_value=False), \
         mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        built = build_image("autoexplore/llava", tmp_path)
        assert built is True
        called = m.call_args.args[0]
        assert called[:3] == ["docker", "build", "-t"]
        assert "autoexplore/llava" in called

def test_build_run_command_passes_gpus_and_mounts():
    cmd = build_run_command(
        image="autoexplore/llava", gpus="2,5",
        mounts=[("/host/run", "/work")], inner_cmd="python infer.py",
    )
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--gpus" in cmd
    gi = cmd.index("--gpus")
    assert cmd[gi + 1] == '"device=2,5"'
    assert "-v" in cmd
    vi = cmd.index("-v")
    assert cmd[vi + 1] == "/host/run:/work"
    # 镜像名后接 sh -c <inner_cmd>
    assert cmd[-4:] == ["autoexplore/llava", "sh", "-c", "python infer.py"]

def test_run_in_container_returns_exit_code(tmp_path):
    log = tmp_path / "run.log"
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        rc = run_in_container(
            image="img", gpus="0", mounts=[], inner_cmd="true", log_path=log,
        )
        assert rc == 0
        assert log.exists()

def test_check_runtime_raises_without_docker():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(DockerError, match="找不到 docker"):
            check_runtime()
