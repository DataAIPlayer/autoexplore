import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import phase2_state as ps


def test_is_significant_relative_5pct():
    assert ps.is_significant(0.42, 0.40) is True      # +5.0% 恰好达门
    assert ps.is_significant(0.4199, 0.40) is False    # +4.97% 不够
    assert ps.is_significant(0.50, 0.40) is True
    assert ps.is_significant(0.40, 0.40) is False      # 持平不晋升


def test_is_significant_zero_backbone():
    assert ps.is_significant(0.01, 0.0) is True
    assert ps.is_significant(0.0, 0.0) is False


def test_best_ready_from_results_picks_highest_ready(toy_run):
    model, metric = ps.best_ready_from_results(toy_run / "results.tsv")
    assert model == "toy-a"
    assert metric == pytest.approx(0.40)


def test_best_ready_raises_when_no_ready(tmp_path):
    f = tmp_path / "results.tsv"
    f.write_text("model\tprimary_metric\tmemory_gb\tstatus\tdescription\n"
                 "x\t0.0\t0.0\tcrash\tnope\n")
    with pytest.raises(ValueError):
        ps.best_ready_from_results(f)


def test_init_state_sets_backbone_from_best_ready(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    assert state["backbone"]["id"] == "p1:toy-a"
    assert state["backbone"]["primary_metric"] == pytest.approx(0.40)
    assert state["backbone"]["version_n"] == 0
    assert state["inference_tuning"] == "pending"
    assert state["last_diagnosed_version"] == -1
    # 落盘且可重新加载
    loaded = ps.load_state(toy_run)
    assert loaded["backbone"]["id"] == "p1:toy-a"


def test_promote_backbone_bumps_version_and_resets_tuning(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    ps.mark_diagnosed(toy_run, state)
    ps.set_inference_tuning(toy_run, state, "explored")
    state = ps.promote_backbone(toy_run, state, "r001:exp_a",
                                "rounds/r001/exp_a", 0.50, {"score_mean": 0.50})
    assert state["backbone"]["version_n"] == 1
    assert state["backbone"]["id"] == "r001:exp_a"
    assert state["backbone"]["primary_metric"] == pytest.approx(0.50)
    assert state["inference_tuning"] == "pending"      # 新主干重评便宜调优
    assert len(state["backbone_history"]) == 2


def test_directions_dedup(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    d = {"title": "Foo Method", "source_urls": ["http://arxiv.org/abs/1"]}
    assert ps.directions_seen(state, d) is False
    ps.directions_add(toy_run, state, [d])
    assert ps.directions_seen(state, d) is True
    # 顺序无关的指纹
    d2 = {"title": "Foo Method", "source_urls": ["http://arxiv.org/abs/1"]}
    assert ps.directions_seen(state, d2) is True


def test_save_state_is_atomic(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    state["round_counter"] = 7
    ps.save_state(toy_run, state)
    assert not (toy_run / "phase2" / "state.json.tmp").exists()  # 临时文件已替换
    assert ps.load_state(toy_run)["round_counter"] == 7


def test_plan_dispatch_non_training_first_training_queues():
    exps = [
        {"slot": "a", "needs_gpus": 1, "is_training": True},
        {"slot": "b", "needs_gpus": 1, "is_training": False},
        {"slot": "c", "needs_gpus": 1, "is_training": False},
    ]
    plan = ps.plan_dispatch(exps, free_gpus=[0, 1])  # 只够两个非训练
    assert set(plan["assigned"].keys()) == {"b", "c"}  # 非训练优先拿卡
    assert plan["queued"] == ["a"]                      # 训练型排队


def test_plan_dispatch_multi_gpu_need():
    exps = [{"slot": "a", "needs_gpus": 3, "is_training": True},
            {"slot": "b", "needs_gpus": 1, "is_training": False}]
    plan = ps.plan_dispatch(exps, free_gpus=[0, 1])
    assert plan["assigned"] == {"b": [0]}   # a 需 3 卡但只剩 1,排队
    assert plan["queued"] == ["a"]


def test_round_open_and_record_lifecycle(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    dirs = [{"slot": "a", "tier": "config", "title": "t1", "source_urls": []},
            {"slot": "b", "tier": "train", "title": "t2", "source_urls": []},
            {"slot": "c", "tier": "pipeline", "title": "t3", "source_urls": []}]
    state = ps.open_round(toy_run, state, dirs)
    assert state["rounds"][-1]["id"] == "r001"
    assert state["rounds"][-1]["status"] == "open"
    assert len(state["directions_tried"]) == 3  # open 同时登记去重
    # 记录前两个 slot → running
    state = ps.record_slot(toy_run, state, "r001", "a", 0.50, "done")
    state = ps.record_slot(toy_run, state, "r001", "b", 0.0, "crash")
    assert state["rounds"][-1]["status"] == "running"
    # delta% 相对主干 0.40
    slot_a = next(s for s in state["rounds"][-1]["slots"] if s["slot"] == "a")
    assert slot_a["delta_pct"] == pytest.approx(25.0)
    # 记录最后一个 → scored
    state = ps.record_slot(toy_run, state, "r001", "c", 0.41, "done")
    assert state["rounds"][-1]["status"] == "scored"


def test_record_slot_unknown_slot_raises(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    state = ps.open_round(toy_run, state, [{"slot": "a", "tier": "config",
                                            "title": "t", "source_urls": []}])
    with pytest.raises(KeyError):
        ps.record_slot(toy_run, state, "r001", "zzz", 0.5, "done")


def test_close_round(toy_run):
    state = ps.init_state(toy_run, "toy-tag")
    state = ps.open_round(toy_run, state, [{"slot": "a", "tier": "config",
                                            "title": "t", "source_urls": []}])
    state = ps.close_round(toy_run, state, "r001")
    assert state["rounds"][-1]["status"] == "done"


def test_next_action_sequence(toy_run):
    # 刚 init:未诊断 → diagnose
    state = ps.init_state(toy_run, "toy-tag")
    assert ps.next_action(state)["action"] == "diagnose"
    # 诊断后:便宜调优待定 → infer_tune
    ps.mark_diagnosed(toy_run, state)
    assert ps.next_action(state)["action"] == "infer_tune"
    # 调优探索完 → search(无 round)
    ps.set_inference_tuning(toy_run, state, "explored")
    assert ps.next_action(state)["action"] == "search"
    # 开了 round 未跑完 → execute
    state = ps.open_round(toy_run, state, [{"slot": "a", "tier": "config",
                                            "title": "t", "source_urls": []}])
    assert ps.next_action(state)["action"] == "execute"
    # 全部 scored → promote_check
    state = ps.record_slot(toy_run, state, "r001", "a", 0.41, "done")
    assert ps.next_action(state)["action"] == "promote_check"
    # 关闭 round → 回 search
    state = ps.close_round(toy_run, state, "r001")
    assert ps.next_action(state)["action"] == "search"


def test_next_action_init_when_no_backbone():
    assert ps.next_action({})["action"] == "init"


def test_append_phase2_result_schema(toy_run):
    ps.append_phase2_result(toy_run, "r001", "exp_a", 0, 0.45, 12.5, "keep", "config tweak")
    ps.append_phase2_result(toy_run, "r001", "exp_b", 0, 0.0, None, "crash", "OOM")
    import csv as _csv
    rows = list(_csv.reader((toy_run / "phase2" / "results.tsv").open(), delimiter="\t"))
    assert rows[0] == ["round", "exp", "base_backbone_ver", "primary_metric",
                       "delta_pct", "status", "description"]
    assert rows[1][0] == "r001" and rows[1][1] == "exp_a"
    assert rows[1][4] == "12.50" and rows[1][5] == "keep"
    assert rows[2][4] == "" and rows[2][5] == "crash"   # delta_pct=None → 空


def test_cli_init_and_resume(toy_run):
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "init",
                        "--run-dir", str(toy_run), "--tag", "toy-tag"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert (toy_run / "phase2" / "state.json").exists()
    r2 = subprocess.run([sys.executable, "scripts/phase2_state.py", "resume",
                         "--run-dir", str(toy_run)], capture_output=True, text=True)
    assert r2.returncode == 0
    assert json.loads(r2.stdout)["action"] == "diagnose"


def test_cli_gate(toy_run):
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "gate",
                        "--candidate", "0.42", "--backbone", "0.40"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert json.loads(r.stdout)["promote"] is True
    r2 = subprocess.run([sys.executable, "scripts/phase2_state.py", "gate",
                         "--candidate", "0.41", "--backbone", "0.40"],
                        capture_output=True, text=True)
    assert json.loads(r2.stdout)["promote"] is False


def test_cli_dispatch(toy_run):
    exps = json.dumps([{"slot": "a", "needs_gpus": 1, "is_training": True},
                       {"slot": "b", "needs_gpus": 1, "is_training": False}])
    r = subprocess.run([sys.executable, "scripts/phase2_state.py", "dispatch",
                        "--experiments", exps, "--free-gpus", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["assigned"] == {"b": [0]}
    assert out["queued"] == ["a"]
