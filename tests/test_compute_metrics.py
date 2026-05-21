import json
import pytest
from scripts.compute_metrics import compute_metrics, MetricsError

TOY_EVAL = '''
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
n = sum(1 for _ in open(a.predictions))
json.dump({"primary_metric": n / 10.0, "metrics": {"count": n}}, open(a.out, "w"))
'''

BAD_EVAL = '''
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--predictions"); ap.add_argument("--dataset"); ap.add_argument("--out")
a = ap.parse_args()
json.dump({"metrics": {}}, open(a.out, "w"))  # 缺 primary_metric
'''

CRASH_EVAL = 'raise SystemExit("eval boom")'

def _setup(tmp_path, eval_src, n_preds=9):
    (tmp_path / "evaluate.py").write_text(eval_src)
    (tmp_path / "dataset").mkdir()
    preds = tmp_path / "predictions.jsonl"
    preds.write_text("".join('{"id": %d}\n' % i for i in range(n_preds)))
    return tmp_path / "evaluate.py", preds, tmp_path / "dataset"

def test_compute_returns_normalized_metrics(tmp_path):
    eval_py, preds, ds = _setup(tmp_path, TOY_EVAL)
    out = tmp_path / "metrics.json"
    result = compute_metrics(eval_py, preds, ds, out)
    assert result["primary_metric"] == pytest.approx(0.9)
    assert result["metrics"]["count"] == 9
    assert json.loads(out.read_text())["primary_metric"] == pytest.approx(0.9)

def test_missing_primary_metric_raises(tmp_path):
    eval_py, preds, ds = _setup(tmp_path, BAD_EVAL)
    with pytest.raises(MetricsError, match="primary_metric"):
        compute_metrics(eval_py, preds, ds, tmp_path / "m.json")

def test_eval_crash_raises(tmp_path):
    eval_py, preds, ds = _setup(tmp_path, CRASH_EVAL)
    with pytest.raises(MetricsError, match="evaluate.py 执行失败"):
        compute_metrics(eval_py, preds, ds, tmp_path / "m.json")
