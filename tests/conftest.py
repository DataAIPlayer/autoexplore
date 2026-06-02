"""共享测试夹具:造一个无 Docker/GPU 依赖的玩具 run。"""
import json

import pytest

# 玩具 evaluate.py:遵守冻结 CLI 契约,从 prediction 的 score 字段取分求均值。
TOY_EVALUATE = '''\
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    preds = {}
    with a.predictions.open() as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                preds[o["image_id"]] = o
    meta = json.loads((a.dataset / "metadata.json").read_text())
    ids = [s["image_id"] for s in meta["samples"]]
    scores, n_skipped = [], 0
    for sid in ids:
        p = preds.get(sid)
        if p is None:
            n_skipped += 1
            scores.append(0.0)
        else:
            scores.append(float(p.get("score", 0.0)))
    primary = sum(scores) / len(scores) if scores else 0.0
    out = {"primary_metric": primary,
           "metrics": {"score_mean": primary, "n_samples": len(ids) - n_skipped,
                       "n_skipped": n_skipped}}
    a.out.write_text(json.dumps(out))
    print(f"primary_metric={primary:.6f}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def toy_run(tmp_path):
    """返回 run_dir。含 results.tsv(两个 ready 模型)、dataset(3 样本,带 group 字段)、玩具 evaluate.py。"""
    run = tmp_path / "runs" / "toy-tag"
    ds = run / "dataset"
    ds.mkdir(parents=True)
    samples = [
        {"image_id": "s1", "group": "A"},
        {"image_id": "s2", "group": "A"},
        {"image_id": "s3", "group": "B"},
    ]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    for s in samples:
        d = ds / s["image_id"]
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"image_id": s["image_id"]}))
    (run / "evaluate.py").write_text(TOY_EVALUATE)
    (run / "results.tsv").write_text(
        "model\tprimary_metric\tmemory_gb\tstatus\tdescription\n"
        "toy-a\t0.400000\t10.0\tready\tbest baseline\n"
        "toy-b\t0.300000\t8.0\tready\trunner up\n"
        "toy-c\t0.000000\t0.0\tcrash\tfailed\n"
    )
    return run
