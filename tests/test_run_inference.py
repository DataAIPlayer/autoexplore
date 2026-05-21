from unittest import mock
from pathlib import Path
from scripts.run_inference import run_inference, INNER_PREDICTIONS_PATH

def test_run_inference_mounts_run_dir_and_returns_predictions(tmp_path):
    run_dir = tmp_path
    (run_dir / "dataset").mkdir()
    with mock.patch("scripts.run_inference.run_in_container", return_value=0) as m:
        preds = run_inference(
            image="autoexplore/llava", gpus="1",
            run_dir=run_dir, infer_cmd="python /work/infer.py",
            log_path=run_dir / "run.log",
        )
        assert preds == run_dir / "predictions.jsonl"
        kwargs = m.call_args.kwargs
        assert kwargs["image"] == "autoexplore/llava"
        assert kwargs["gpus"] == "1"
        assert (str(run_dir), "/work") in kwargs["mounts"]
        # 推理命令应把输出写到约定路径
        assert INNER_PREDICTIONS_PATH in kwargs["inner_cmd"]

def test_run_inference_raises_on_nonzero(tmp_path):
    (tmp_path / "dataset").mkdir()
    with mock.patch("scripts.run_inference.run_in_container", return_value=1):
        import pytest
        with pytest.raises(RuntimeError, match="推理失败"):
            run_inference(
                image="img", gpus="0", run_dir=tmp_path,
                infer_cmd="false", log_path=tmp_path / "run.log",
            )
