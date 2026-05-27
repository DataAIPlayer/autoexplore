import pytest
from scripts.gpu_select import parse_nvidia_smi, select_gpus, GpuInfo

SAMPLE = "0, 81920, 80000\n1, 81920, 200\n2, 81920, 150\n3, 81920, 40000\n"

def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi(SAMPLE)
    assert gpus == [
        GpuInfo(index=0, total_mib=81920, used_mib=80000),
        GpuInfo(index=1, total_mib=81920, used_mib=200),
        GpuInfo(index=2, total_mib=81920, used_mib=150),
        GpuInfo(index=3, total_mib=81920, used_mib=40000),
    ]

def test_parse_ignores_blank_lines():
    assert parse_nvidia_smi("0, 100, 10\n\n") == [GpuInfo(0, 100, 10)]

def test_select_picks_most_free_first():
    gpus = parse_nvidia_smi(SAMPLE)
    # 空闲: gpu2=81770, gpu1=81720, gpu3=41920, gpu0=1920
    assert select_gpus(gpus, count=2) == [2, 1]

def test_select_respects_min_free_mib():
    gpus = parse_nvidia_smi(SAMPLE)
    # 只有 gpu1,gpu2 满足 >=50000 空闲
    assert select_gpus(gpus, count=4, min_free_mib=50000) == [2, 1]

def test_select_raises_when_no_gpu_meets_threshold():
    gpus = parse_nvidia_smi("0, 100, 99\n")
    with pytest.raises(RuntimeError, match="无空闲 GPU"):
        select_gpus(gpus, count=1, min_free_mib=50)
