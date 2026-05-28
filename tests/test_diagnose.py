import json
import sys
from pathlib import Path

import pytest

from scripts import diagnose
from tests.helpers import write_predictions


def test_make_view_subsets_metadata_and_symlinks(toy_run, tmp_path):
    view = tmp_path / "view"
    diagnose._make_view(toy_run / "dataset", ["s1", "s3"], view)
    meta = json.loads((view / "metadata.json").read_text())
    assert [s["image_id"] for s in meta["samples"]] == ["s1", "s3"]
    assert (view / "s1").exists() and (view / "s3").exists()
    assert not (view / "s2").exists()


def test_detect_group_fields(toy_run):
    samples = json.loads((toy_run / "dataset" / "metadata.json").read_text())["samples"]
    assert diagnose._detect_group_fields(samples) == ["group"]


def test_detect_group_fields_degrades_when_all_unique(tmp_path):
    samples = [{"image_id": f"s{i}", "uid": f"u{i}"} for i in range(4)]
    # uid 全不同(4 distinct = n),超过 0.5*n 阈值 → 不分组
    assert diagnose._detect_group_fields(samples) == []


def test_diagnose_full_per_sample_worst_and_groups(toy_run, tmp_path):
    preds = toy_run / "predictions" / "predictions.jsonl"
    write_predictions(preds, {"s1": 0.9, "s2": 0.1, "s3": 0.5})  # group A=[.9,.1], B=[.5]
    work = tmp_path / "diagwork"
    out = diagnose.diagnose(preds, toy_run / "dataset", toy_run / "evaluate.py",
                            worst_k=2, work_dir=work)
    # full = mean(.9,.1,.5) = .5
    assert out["full"]["primary_metric"] == pytest.approx(0.5)
    # per-sample 升序,worst_k=2 → s2(.1), s3(.5)
    assert [w["id"] for w in out["worst_k"]] == ["s2", "s3"]
    # 分组:A=mean(.9,.1)=.5, B=.5
    assert out["groups"]["group"]["A"] == pytest.approx(0.5)
    assert out["groups"]["group"]["B"] == pytest.approx(0.5)


def test_diagnose_handles_messy_group_values(toy_run, tmp_path):
    # 分组值含 '/' 和空格,不应产生嵌套目录或崩溃,且真实值作为 key 保留
    ds = tmp_path / "ds"
    ds.mkdir()
    samples = [{"image_id": "s1", "fmt": "a/b"},
               {"image_id": "s2", "fmt": "c d"}]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    for s in samples:
        (ds / s["image_id"]).mkdir()
        (ds / s["image_id"] / "meta.json").write_text(json.dumps({"image_id": s["image_id"]}))
    preds = tmp_path / "preds.jsonl"
    write_predictions(preds, {"s1": 0.8, "s2": 0.2})
    out = diagnose.diagnose(preds, ds, toy_run / "evaluate.py",
                            worst_k=2, work_dir=tmp_path / "w")
    assert out["groups"]["fmt"]["a/b"] == pytest.approx(0.8)
    assert out["groups"]["fmt"]["c d"] == pytest.approx(0.2)


def test_diagnose_cli_writes_json_and_md(toy_run, tmp_path):
    preds = toy_run / "predictions" / "predictions.jsonl"
    write_predictions(preds, {"s1": 0.9, "s2": 0.1, "s3": 0.5})
    out_json = tmp_path / "diag.json"
    out_md = tmp_path / "diag.md"
    import subprocess
    r = subprocess.run([sys.executable, "scripts/diagnose.py",
                        "--predictions", str(preds),
                        "--dataset", str(toy_run / "dataset"),
                        "--evaluate-py", str(toy_run / "evaluate.py"),
                        "--out-json", str(out_json), "--out-md", str(out_md),
                        "--worst-k", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out_json.read_text())["full"]["primary_metric"] == pytest.approx(0.5)
    assert "worst" in out_md.read_text().lower()
