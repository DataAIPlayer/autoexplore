import json
import csv
from pathlib import Path

import pytest

from scripts import phase2_state as ps
from scripts import diagnose
from tests.helpers import write_predictions


def test_full_round_file_flow(toy_run):
    # 0. init 主干
    state = ps.init_state(toy_run, "toy-tag")
    assert ps.next_action(state)["action"] == "diagnose"

    # 1. 诊断当前主干(用一份 backbone predictions)
    bb_preds = toy_run / "phase2" / "backbone_preds.jsonl"
    write_predictions(bb_preds, {"s1": 0.5, "s2": 0.2, "s3": 0.41})
    diag = diagnose.diagnose(bb_preds, toy_run / "dataset", toy_run / "evaluate.py",
                             worst_k=2, work_dir=toy_run / "phase2" / "diagwork")
    assert diag["full"]["primary_metric"] == pytest.approx(0.37, abs=1e-6)
    ps.mark_diagnosed(toy_run, state)
    ps.set_inference_tuning(toy_run, state, "explored")
    assert ps.next_action(state)["action"] == "search"

    # 2. 开一轮 3 方向
    dirs = [{"slot": "a", "tier": "config", "title": "A", "source_urls": ["u1"]},
            {"slot": "b", "tier": "pipeline", "title": "B", "source_urls": ["u2"]},
            {"slot": "c", "tier": "train", "title": "C", "source_urls": ["u3"]}]
    state = ps.open_round(toy_run, state, dirs)
    rid = state["rounds"][-1]["id"]

    # 3. 模拟 3 个 exp 推理+评分(写 predictions → 真跑玩具 evaluate via diagnose.full 路径)
    #    这里直接用 evaluate.py 算每个 exp 的 metrics.json,模拟 compute_metrics 产物。
    import subprocess, sys
    exp_scores = {"a": {"s1": 0.45, "s2": 0.45, "s3": 0.45},   # mean .45 (+12.5%)
                  "b": {"s1": 0.40, "s2": 0.40, "s3": 0.40},   # mean .40 (持平)
                  "c": {"s1": 0.10, "s2": 0.10, "s3": 0.10}}   # mean .10
    for slot, scores in exp_scores.items():
        exp_dir = toy_run / "phase2" / "rounds" / rid / f"exp_{slot}"
        preds = exp_dir / "predictions.jsonl"
        write_predictions(preds, scores)
        metrics_json = exp_dir / "metrics.json"
        subprocess.run([sys.executable, str(toy_run / "evaluate.py"),
                        "--predictions", str(preds),
                        "--dataset", str(toy_run / "dataset"),
                        "--out", str(metrics_json)], check=True)
        m = json.loads(metrics_json.read_text())
        state = ps.record_slot(toy_run, state, rid, slot, m["primary_metric"], "done")

    assert state["rounds"][-1]["status"] == "scored"
    assert ps.next_action(state)["action"] == "promote_check"

    # 4. 晋升判定:最佳是 a=.45 vs 主干 .40 → +12.5% ≥ 5% → 晋升
    slots = state["rounds"][-1]["slots"]
    best = max(slots, key=lambda s: s["primary_metric"])
    assert best["slot"] == "a"
    assert ps.is_significant(best["primary_metric"], state["backbone"]["primary_metric"])
    base_ver = state["backbone"]["version_n"]  # 晋升前的主干版本(记录到 results 行)
    state = ps.promote_backbone(toy_run, state, f"{rid}:exp_a",
                                f"rounds/{rid}/exp_a", best["primary_metric"],
                                {"score_mean": best["primary_metric"]})
    ps.append_phase2_result(toy_run, rid, "exp_a", base_ver, best["primary_metric"],
                            best["delta_pct"], "keep", "promoted from config tweak")
    state = ps.close_round(toy_run, state, rid)

    # 5. 断言全链路 schema + 状态
    assert state["backbone"]["version_n"] == 1
    assert state["backbone"]["primary_metric"] == pytest.approx(0.45)
    assert state["inference_tuning"] == "pending"          # 新主干重置便宜调优
    assert ps.next_action(state)["action"] == "diagnose"   # 回到诊断,形成螺旋

    # results.tsv schema(phase-2 7 列)
    rows = list(csv.reader((toy_run / "phase2" / "results.tsv").open(), delimiter="\t"))
    assert rows[0] == ["round", "exp", "base_backbone_ver", "primary_metric",
                       "delta_pct", "status", "description"]
    assert rows[1][5] == "keep"
    assert rows[1][4] == "12.50"   # delta% 相对晋升前主干 0.40

    # state.json 落盘且字段齐全
    loaded = ps.load_state(toy_run)
    assert loaded["backbone"]["id"] == f"{rid}:exp_a"
    assert loaded["directions_tried"]  # 3 方向已登记
