import json
import time
from pathlib import Path

from scripts import train_launch as tl


def test_latest_checkpoint_picks_newest(tmp_path):
    ck = tmp_path / "ckpts"
    ck.mkdir()
    a = ck / "step100.pt"; a.write_text("x")
    time.sleep(0.01)
    b = ck / "step200.pt"; b.write_text("y")
    assert tl.latest_checkpoint(ck) == b


def test_latest_checkpoint_none_when_empty(tmp_path):
    assert tl.latest_checkpoint(tmp_path / "nope") is None


def test_record_provenance(tmp_path):
    tl.record_provenance(tmp_path, "CrelloTrain", "modelscope", "caches/modelscope/x")
    p = json.loads((tmp_path / "dataset_provenance.json").read_text())
    assert p["dataset"] == "CrelloTrain"
    assert p["source"] == "modelscope"


def test_build_launch_cmd_multi_gpu_and_resume(tmp_path):
    train_py = tmp_path / "train.py"
    train_py.write_text("")
    cmd = tl.build_launch_cmd(train_py, gpus=[0, 1, 2], extra_args=["--lr", "1e-4"],
                              resume_from=tmp_path / "ckpts" / "step200.pt")
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=3" in cmd
    assert "--lr" in cmd and "1e-4" in cmd
    assert "--resume" in cmd


def test_build_launch_cmd_no_resume(tmp_path):
    train_py = tmp_path / "train.py"; train_py.write_text("")
    cmd = tl.build_launch_cmd(train_py, gpus=[0], extra_args=[], resume_from=None)
    assert "--resume" not in cmd


def test_run_with_budget_times_out(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["sleep", "5"], log, gpus=[0], budget_seconds=1)
    assert status == "timeout"


def test_run_with_budget_done(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["true"], log, gpus=[0], budget_seconds=5)
    assert status == "done"


def test_run_with_budget_error_on_nonzero_exit(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["false"], log, gpus=[0], budget_seconds=5)
    assert status == "error"


def test_run_with_budget_error_when_launcher_missing(tmp_path):
    log = tmp_path / "train.log"
    status = tl.run_with_budget(["no_such_binary_xyz"], log, gpus=[0], budget_seconds=5)
    assert status == "error"           # 启动器缺失不崩溃编排器
    assert "not found" in log.read_text().lower()
