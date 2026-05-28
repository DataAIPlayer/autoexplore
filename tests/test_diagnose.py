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


def test_detect_id_field_picks_subdir_matching_unique_key(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    samples = [{"uid": "u1", "bucket": "x"}, {"uid": "u2", "bucket": "y"}]
    for s in samples:
        (ds / s["uid"]).mkdir()
    # bucket 值唯一但不对应子目录;uid 唯一且对应子目录 → 选 uid
    assert diagnose._detect_id_field(samples, ds) == "uid"


def test_diagnose_generic_id_field(tmp_path):
    # 数据集用 'uid' 而非 image_id 标识样本,diagnose 应自动探测,不绑死字段名
    ds = tmp_path / "ds"
    ds.mkdir()
    samples = [{"uid": "u1", "bucket": "x"}, {"uid": "u2", "bucket": "y"}]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    for s in samples:
        (ds / s["uid"]).mkdir()
        (ds / s["uid"] / "meta.json").write_text(json.dumps({"uid": s["uid"]}))
    ev = tmp_path / "evaluate.py"
    ev.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--predictions', type=Path)\n"
        "ap.add_argument('--dataset', type=Path)\n"
        "ap.add_argument('--out', type=Path)\n"
        "a = ap.parse_args()\n"
        "preds = {}\n"
        "for line in a.predictions.open():\n"
        "    line = line.strip()\n"
        "    if line:\n"
        "        o = json.loads(line); preds[o['uid']] = o\n"
        "ids = [s['uid'] for s in json.loads((a.dataset/'metadata.json').read_text())['samples']]\n"
        "sc = [float(preds.get(i, {}).get('score', 0.0)) for i in ids]\n"
        "p = sum(sc)/len(sc) if sc else 0.0\n"
        "a.out.write_text(json.dumps({'primary_metric': p, 'metrics': {'n': len(ids)}}))\n"
    )
    preds = tmp_path / "p.jsonl"
    with preds.open("w") as f:
        f.write(json.dumps({"uid": "u1", "score": 0.8}) + "\n")
        f.write(json.dumps({"uid": "u2", "score": 0.2}) + "\n")
    out = diagnose.diagnose(preds, ds, ev, worst_k=2, work_dir=tmp_path / "w")
    assert [w["id"] for w in out["worst_k"]] == ["u2", "u1"]   # 自动用 uid 逐样本
    assert out["groups"]["bucket"]["x"] == pytest.approx(0.8)  # 分组排除 uid


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
