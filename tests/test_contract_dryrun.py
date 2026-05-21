"""端到端文件流转契约:candidates → predictions → metrics → results。"""
import json
from pathlib import Path
from scripts.compute_metrics import compute_metrics
from scripts.progress import (
    Progress, save_progress, load_progress, is_done,
    append_result, read_results,
)

TOY_EVAL = '''
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--predictions"); ap.add_argument("--dataset"); ap.add_argument("--out")
a = ap.parse_args()
preds = [json.loads(l) for l in open(a.predictions)]
correct = sum(1 for p in preds if p["pred"] == p["gold"])
acc = correct / len(preds)
json.dump({"primary_metric": acc, "metrics": {"accuracy": acc, "n": len(preds)}},
          open(a.out, "w"))
'''

def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "toy"
    (run / "dataset").mkdir(parents=True)
    (run / "evaluate.py").write_text(TOY_EVAL)
    candidates = [
        {"name": "model-a", "repo": "x/a", "reported_score": 0.9, "priority": 1},
        {"name": "model-b", "repo": "x/b", "reported_score": 0.8, "priority": 2},
    ]
    (run / "candidates.json").write_text(json.dumps(candidates))
    return run

def _stub_inference(model_dir: Path, n_correct: int, n_total: int) -> Path:
    """替代真实容器推理:直接写 predictions.jsonl。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    preds = model_dir / "predictions.jsonl"
    lines = []
    for i in range(n_total):
        gold = "yes"
        pred = "yes" if i < n_correct else "no"
        lines.append(json.dumps({"id": i, "pred": pred, "gold": gold}))
    preds.write_text("\n".join(lines) + "\n")
    return preds

def test_full_phase1_file_flow(tmp_path):
    run = _make_run(tmp_path)
    candidates = json.loads((run / "candidates.json").read_text())
    assert [c["name"] for c in candidates] == ["model-a", "model-b"]

    for c in sorted(candidates, key=lambda x: x["priority"]):
        model_dir = run / "models" / c["name"]
        save_progress(model_dir, Progress(model=c["name"], stage="A"))

        # 模拟复现成功 + stub 推理
        n_correct = 8 if c["name"] == "model-a" else 6
        preds = _stub_inference(model_dir, n_correct=n_correct, n_total=10)
        save_progress(model_dir, Progress(model=c["name"], stage="C"))

        # 真实 compute_metrics 接 stub predictions
        metrics = compute_metrics(
            run / "evaluate.py", preds, run / "dataset",
            model_dir / "metrics.json",
        )
        save_progress(model_dir, Progress(model=c["name"], stage="ready"))
        assert is_done(model_dir)
        append_result(
            run, c["name"], metrics["primary_metric"],
            memory_gb=0.0, status="ready",
            description=f"reproduced {c['repo']}",
        )

    rows = read_results(run)
    assert rows[0] == ["model", "primary_metric", "memory_gb", "status", "description"]
    data = {r[0]: float(r[1]) for r in rows[1:]}
    assert data == {"model-a": 0.8, "model-b": 0.6}
    # 步骤 10:选 primary_metric 最高者
    best = max(data, key=data.get)
    assert best == "model-a"

def test_crash_does_not_block_other_models(tmp_path):
    run = _make_run(tmp_path)
    # model-a 复现失败
    append_result(run, "model-a", 0.0, 0.0, "crash", "OOM")
    # model-b 成功
    mb = run / "models" / "model-b"
    preds = _stub_inference(mb, n_correct=7, n_total=10)
    compute_metrics(run / "evaluate.py", preds, run / "dataset", mb / "metrics.json")
    append_result(run, "model-b", 0.7, 0.0, "ready", "ok")

    rows = read_results(run)
    statuses = {r[0]: r[3] for r in rows[1:]}
    assert statuses == {"model-a": "crash", "model-b": "ready"}
