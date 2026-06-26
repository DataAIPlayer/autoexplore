# tests/test_smoke_phase3_gpu.py
"""真实环境冒烟:需 GPU + 真框架,默认 skip。CI/无卡机器不跑。
手动:uv run pytest tests/test_smoke_phase3_gpu.py -m gpu --no-header"""
import shutil

import pytest

pytestmark = pytest.mark.gpu


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="需要可用 CUDA GPU")
def test_benchmark_measures_real_latency(tmp_path):
    """最小真实 adapter(torch matmul)走 benchmark.measure,产出非零延迟与 CUDA sync。"""
    import json
    from pathlib import Path
    from scripts import benchmark as bm

    ds = tmp_path / "dataset"
    ds.mkdir()
    samples = [{"image_id": f"s{i}"} for i in range(3)]
    (ds / "metadata.json").write_text(json.dumps({"samples": samples}))
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import torch\n"
        "def load_model(config):\n"
        "    return torch.randn(512, 512, device='cuda')\n"
        "def infer_one(handle, record):\n"
        "    y = handle @ handle\n"
        "    return {'image_id': record['image_id'], 'score': float(y.sum().item()) * 0 + 0.4}\n"
    )
    mod = bm.load_adapter(adapter)
    speed, preds = bm.measure(mod, ds, {"warmup": 2, "iters": 3, "n_records": 3,
                                        "gpu_name": "real", "gpu_count": 1})
    assert speed["latency_mean_ms"] > 0
    assert len(preds) == 3
