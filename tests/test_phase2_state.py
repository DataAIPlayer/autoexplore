import json
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
