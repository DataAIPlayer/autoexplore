import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu


def test_gpu_select_feeds_dispatch():
    """真实选卡 → 喂给 plan_dispatch,断言派发不为空。"""
    from scripts import phase2_state as ps
    r = subprocess.run([sys.executable, "scripts/gpu_select.py",
                        "--count", "1", "--min-free-mib", "1000"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no free GPU")
    gpus = [int(x) for x in r.stdout.strip().split(",") if x != ""]
    plan = ps.plan_dispatch([{"slot": "a", "needs_gpus": 1, "is_training": False}], gpus)
    assert plan["assigned"].get("a")


def test_train_launch_runs_minimal_train_py(tmp_path):
    """真起一个最小 train.py(纯 CPU 即可),验证壳能拼命令并跑完。"""
    from scripts import train_launch as tl
    exp = tmp_path / "exp"
    exp.mkdir()
    train_py = exp / "train.py"
    train_py.write_text("import sys; print('trained'); sys.exit(0)\n")
    # 用 python 直跑(不经 torchrun)验证 run_with_budget;torchrun 路径在真环境另测
    status = tl.run_with_budget([sys.executable, str(train_py)],
                                exp / "train.log", gpus=[0], budget_seconds=30)
    assert status == "done"
    assert "trained" in (exp / "train.log").read_text()
