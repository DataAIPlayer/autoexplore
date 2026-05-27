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


def test_run_inference_forwards_dataset_extra_mounts_env_user_runtime(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with mock.patch("scripts.run_inference.run_in_container", return_value=0) as m:
        run_inference(
            image="autoexplore/qwen", gpus="4",
            run_dir=run_dir, infer_cmd="python infer.py",
            log_path=run_dir / "run.log",
            dataset_dir=dataset_dir,
            extra_mounts=[("/host/cache/ms", "/cache/modelscope:ro")],
            env={"HF_HOME": "/cache/huggingface"},
            user="1001:1001",
            runtime="nvidia",
        )
        kwargs = m.call_args.kwargs
        # 基础挂载 + dataset_dir + extra 都要在
        assert (str(run_dir), "/work") in kwargs["mounts"]
        assert (str(dataset_dir), "/work/dataset") in kwargs["mounts"]
        assert ("/host/cache/ms", "/cache/modelscope:ro") in kwargs["mounts"]
        # 新增 kwargs 透传
        assert kwargs["env"] == {"HF_HOME": "/cache/huggingface"}
        assert kwargs["user"] == "1001:1001"
        assert kwargs["runtime"] == "nvidia"


def test_run_inference_defaults_runtime_to_nvidia(tmp_path):
    (tmp_path / "dataset").mkdir()
    with mock.patch("scripts.run_inference.run_in_container", return_value=0) as m:
        run_inference(
            image="img", gpus="0", run_dir=tmp_path,
            infer_cmd="true", log_path=tmp_path / "run.log",
        )
        assert m.call_args.kwargs["runtime"] == "nvidia"
