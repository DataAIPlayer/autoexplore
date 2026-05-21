# autoexplore 第一阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建第一版「开源模型自动复现」agent,覆盖需求文档第一阶段步骤 1–10,形态为 Skill 编排 + Python 工具脚本。

**Architecture:** Python 脚本各管一件确定性重活(选卡 / Docker / 推理 / 算指标 / 进度),通过文件系统约定(工作目录 + JSON schema)与 Skill 通信;SKILL.md + reference 文档引导 Claude 做判断与编排。脚本逻辑可在无 Docker/GPU 的开发机上用 mock/stub 单元测试,真实环境冒烟测试标记后在 GPU 服务器手动跑。

**Tech Stack:** Python 3.12,uv(包管理 + 运行,沿用 autoresearch 示例),pytest(测试),标准库 subprocess/json/argparse;运行时依赖 docker CLI + nvidia-smi(仅服务器)。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `pyproject.toml` | uv 项目定义,pytest 配置(含 `gpu` marker) |
| `scripts/__init__.py` | 标记 scripts 为包,便于测试导入 |
| `scripts/gpu_select.py` | 解析 `nvidia-smi`,按空闲显存挑卡,打印 `CUDA_VISIBLE_DEVICES` |
| `scripts/docker_env.py` | 检查 Docker/runtime、构建/复用镜像、容器内执行命令 |
| `scripts/run_inference.py` | 在容器内对 dataset 跑推理,产出 predictions(每行一条 JSON) |
| `scripts/compute_metrics.py` | 调用 run 的 `evaluate.py`,归一成 metrics.json |
| `scripts/progress.py` | 读写 progress.json、汇总 results.tsv、中断恢复查询 |
| `tests/test_gpu_select.py` | gpu_select 单测(nvidia-smi fixture) |
| `tests/test_progress.py` | progress 单测(tmp 目录假 run) |
| `tests/test_compute_metrics.py` | compute_metrics 单测(玩具 evaluate.py) |
| `tests/test_docker_env.py` | docker_env 单测(mock subprocess) |
| `tests/test_contract_dryrun.py` | 端到端 dry-run 契约测试(锁脚本间 schema) |
| `tests/test_smoke_gpu.py` | 真实 Docker 冒烟测试(`@pytest.mark.gpu`,默认 skip) |
| `SKILL.md` | 流程与决策规则 |
| `references/reproduction-loop.md` | 复现循环细则 |

约定:所有脚本是独立 CLI(`uv run scripts/<name>.py ...`),也可作为模块 import 供测试调用。

---

## Task 1: 项目脚手架与 pytest 配置

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "autoexplore"
version = "0.1.0"
description = "自动复现与优化开源模型的 agent (第一阶段)"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
markers = [
    "gpu: 需真实 Docker + GPU 环境,默认在开发机跳过",
]
addopts = "-m 'not gpu'"
testpaths = ["tests"]
```

- [ ] **Step 2: 创建空包标记文件**

`scripts/__init__.py`:

```python
```

`tests/__init__.py`:

```python
```

- [ ] **Step 3: 验证环境装配**

Run: `uv sync && uv run pytest --collect-only`
Expected: 退出码 0,显示 "collected 0 items"(尚无测试)。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml scripts/__init__.py tests/__init__.py
git commit -m "chore: scaffold uv project and pytest config"
```

---

## Task 2: gpu_select.py — 按空闲度选卡

`nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits` 输出形如:
```
0, 81920, 80000
1, 81920, 200
2, 81920, 150
```
即每行 `index, total_MiB, used_MiB`。空闲显存 = total - used。

**Files:**
- Create: `scripts/gpu_select.py`
- Test: `tests/test_gpu_select.py`

- [ ] **Step 1: 写失败测试**

`tests/test_gpu_select.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_gpu_select.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.gpu_select'`。

- [ ] **Step 3: 实现 gpu_select.py**

```python
"""按显卡空闲显存挑选 GPU,打印 CUDA_VISIBLE_DEVICES 供容器使用。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx, total, used = (p.strip() for p in line.split(","))
        gpus.append(GpuInfo(int(idx), int(total), int(used)))
    return gpus


def select_gpus(gpus: list[GpuInfo], count: int, min_free_mib: int = 0) -> list[int]:
    eligible = [g for g in gpus if g.free_mib >= min_free_mib]
    if not eligible:
        raise RuntimeError(
            f"无空闲 GPU:没有显卡满足空闲显存 >= {min_free_mib} MiB"
        )
    eligible.sort(key=lambda g: g.free_mib, reverse=True)
    return [g.index for g in eligible[:count]]


def query_gpus() -> list[GpuInfo]:
    try:
        out = subprocess.run(
            NVIDIA_SMI_QUERY, capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError as e:
        raise RuntimeError("找不到 nvidia-smi:本机无 NVIDIA runtime") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 执行失败: {e.stderr}") from e
    return parse_nvidia_smi(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按空闲度选 GPU")
    ap.add_argument("--count", type=int, default=1, help="需要的 GPU 数量")
    ap.add_argument("--min-free-mib", type=int, default=0, help="每卡最小空闲显存(MiB)")
    args = ap.parse_args(argv)
    try:
        chosen = select_gpus(query_gpus(), args.count, args.min_free_mib)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(",".join(str(i) for i in chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_gpu_select.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/gpu_select.py tests/test_gpu_select.py
git commit -m "feat(gpu): select GPUs by free memory from nvidia-smi"
```

---

## Task 3: progress.py — 进度与结果汇总

progress.json schema(对齐 spec 第 5 节):`model, stage, retry_count, gpus, image_tag, last_error, updated_at`。stage ∈ {A,B,C,ready,crash}。results.tsv 列:`model  primary_metric  memory_gb  status  description`(制表符分隔)。

**Files:**
- Create: `scripts/progress.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: 写失败测试**

`tests/test_progress.py`:

```python
import json
from scripts.progress import (
    load_progress, save_progress, Progress,
    append_result, read_results, is_done,
)

def test_save_then_load_roundtrip(tmp_path):
    p = Progress(model="llava", stage="C", retry_count=1,
                 gpus="2,5", image_tag="autoexplore/llava",
                 last_error="boom")
    save_progress(tmp_path, p)
    loaded = load_progress(tmp_path)
    assert loaded.model == "llava"
    assert loaded.stage == "C"
    assert loaded.retry_count == 1
    assert loaded.gpus == "2,5"
    assert loaded.last_error == "boom"

def test_save_sets_updated_at(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="A"))
    raw = json.loads((tmp_path / "progress.json").read_text())
    assert raw["updated_at"]  # 非空时间戳

def test_load_missing_returns_none(tmp_path):
    assert load_progress(tmp_path) is None

def test_is_done_true_for_terminal_stages(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="ready"))
    assert is_done(tmp_path) is True
    save_progress(tmp_path, Progress(model="m", stage="crash"))
    assert is_done(tmp_path) is True

def test_is_done_false_for_in_progress(tmp_path):
    save_progress(tmp_path, Progress(model="m", stage="B"))
    assert is_done(tmp_path) is False

def test_is_done_false_when_no_progress(tmp_path):
    assert is_done(tmp_path) is False

def test_append_result_writes_header_once(tmp_path):
    append_result(tmp_path, "llava", 0.91, 44.0, "ready", "baseline")
    append_result(tmp_path, "blip", 0.88, 40.0, "ready", "second model")
    rows = read_results(tmp_path)
    assert rows[0] == ["model", "primary_metric", "memory_gb", "status", "description"]
    assert rows[1] == ["llava", "0.910000", "44.0", "ready", "baseline"]
    assert rows[2] == ["blip", "0.880000", "40.0", "ready", "second model"]

def test_append_result_crash_formatting(tmp_path):
    append_result(tmp_path, "x", 0.0, 0.0, "crash", "OOM")
    rows = read_results(tmp_path)
    assert rows[1] == ["x", "0.000000", "0.0", "crash", "OOM"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.progress'`。

- [ ] **Step 3: 实现 progress.py**

```python
"""复现进度持久化(progress.json)与结果汇总(results.tsv)。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

PROGRESS_FILE = "progress.json"
RESULTS_FILE = "results.tsv"
TERMINAL_STAGES = {"ready", "crash"}
RESULTS_HEADER = ["model", "primary_metric", "memory_gb", "status", "description"]


@dataclass
class Progress:
    model: str
    stage: str  # A/B/C/ready/crash
    retry_count: int = 0
    gpus: str = ""
    image_tag: str = ""
    last_error: str = ""
    updated_at: str = ""


def save_progress(model_dir: Path, progress: Progress) -> None:
    progress.updated_at = datetime.now(timezone.utc).isoformat()
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / PROGRESS_FILE).write_text(
        json.dumps(asdict(progress), ensure_ascii=False, indent=2)
    )


def load_progress(model_dir: Path) -> Progress | None:
    path = model_dir / PROGRESS_FILE
    if not path.exists():
        return None
    return Progress(**json.loads(path.read_text()))


def is_done(model_dir: Path) -> bool:
    p = load_progress(model_dir)
    return p is not None and p.stage in TERMINAL_STAGES


def append_result(
    run_dir: Path, model: str, primary_metric: float,
    memory_gb: float, status: str, description: str,
) -> None:
    path = run_dir / RESULTS_FILE
    is_new = not path.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if is_new:
            w.writerow(RESULTS_HEADER)
        w.writerow([
            model, f"{primary_metric:.6f}", f"{memory_gb:.1f}",
            status, description,
        ])


def read_results(run_dir: Path) -> list[list[str]]:
    path = run_dir / RESULTS_FILE
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="进度与结果工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("set", help="写入 progress.json")
    s.add_argument("--model-dir", type=Path, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--stage", required=True)
    s.add_argument("--retry-count", type=int, default=0)
    s.add_argument("--gpus", default="")
    s.add_argument("--image-tag", default="")
    s.add_argument("--last-error", default="")

    g = sub.add_parser("done", help="查询模型是否处于终态,done 退出 0 否则 1")
    g.add_argument("--model-dir", type=Path, required=True)

    r = sub.add_parser("result", help="追加一行 results.tsv")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--primary-metric", type=float, required=True)
    r.add_argument("--memory-gb", type=float, required=True)
    r.add_argument("--status", required=True)
    r.add_argument("--description", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "set":
        save_progress(args.model_dir, Progress(
            model=args.model, stage=args.stage, retry_count=args.retry_count,
            gpus=args.gpus, image_tag=args.image_tag, last_error=args.last_error,
        ))
        return 0
    if args.cmd == "done":
        return 0 if is_done(args.model_dir) else 1
    if args.cmd == "result":
        append_result(
            args.run_dir, args.model, args.primary_metric,
            args.memory_gb, args.status, args.description,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_progress.py -v`
Expected: 8 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/progress.py tests/test_progress.py
git commit -m "feat(progress): persist reproduction state and results summary"
```

---

## Task 4: compute_metrics.py — 调用 evaluate.py 归一指标

约定:每个 run 的 `evaluate.py` 是独立脚本,被调用方式为
`python evaluate.py --predictions <path> --dataset <dir> --out <metrics.json>`,
它写出 `{"primary_metric": <float>, "metrics": {...}}`。compute_metrics 负责调用它、校验输出 schema,失败则抛错(由上层标 crash)。

**Files:**
- Create: `scripts/compute_metrics.py`
- Test: `tests/test_compute_metrics.py`

- [ ] **Step 1: 写失败测试**

`tests/test_compute_metrics.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_compute_metrics.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.compute_metrics'`。

- [ ] **Step 3: 实现 compute_metrics.py**

```python
"""调用 run 专属的 evaluate.py,校验并归一成 metrics.json。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class MetricsError(RuntimeError):
    """评测失败:evaluate.py 崩溃或输出 schema 不合法。"""


def compute_metrics(
    evaluate_py: Path, predictions: Path, dataset_dir: Path, out: Path
) -> dict:
    cmd = [
        sys.executable, str(evaluate_py),
        "--predictions", str(predictions),
        "--dataset", str(dataset_dir),
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MetricsError(f"evaluate.py 执行失败: {proc.stderr.strip()}")
    if not out.exists():
        raise MetricsError("evaluate.py 未写出 metrics 文件")
    data = json.loads(out.read_text())
    if "primary_metric" not in data:
        raise MetricsError("metrics.json 缺少 primary_metric 字段")
    if not isinstance(data["primary_metric"], (int, float)):
        raise MetricsError("primary_metric 必须是数字")
    data.setdefault("metrics", {})
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="计算评价指标")
    ap.add_argument("--evaluate-py", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = compute_metrics(
            args.evaluate_py, args.predictions, args.dataset, args.out
        )
    except MetricsError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"primary_metric: {result['primary_metric']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_compute_metrics.py -v`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_metrics.py tests/test_compute_metrics.py
git commit -m "feat(metrics): invoke run evaluate.py and normalize metrics.json"
```

---

## Task 5: docker_env.py — 环境检查 / 镜像构建复用 / 容器执行

职责:(1) `check()` 验证 docker + nvidia runtime 可用,fail fast;(2) `image_exists(tag)` 查镜像是否已存在;(3) `build(tag, dockerfile_dir)` 不存在才构建;(4) `run_in_container(...)` 以指定 GPU 跑命令,输出重定向到 log 文件。全部通过拼装 docker CLI 命令实现,测试用 mock subprocess,不真起容器。

**Files:**
- Create: `scripts/docker_env.py`
- Test: `tests/test_docker_env.py`

- [ ] **Step 1: 写失败测试**

`tests/test_docker_env.py`:

```python
from unittest import mock
import subprocess
import pytest
from scripts.docker_env import (
    image_exists, build_image, run_in_container,
    check_runtime, DockerError, build_run_command,
)

def test_image_exists_true_when_inspect_succeeds():
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        assert image_exists("autoexplore/llava") is True
        m.assert_called_once()
        assert "image" in m.call_args.args[0]
        assert "inspect" in m.call_args.args[0]

def test_image_exists_false_when_inspect_fails():
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 1, "", "No such image")
        assert image_exists("missing") is False

def test_build_image_skips_when_exists(tmp_path):
    with mock.patch("scripts.docker_env.image_exists", return_value=True), \
         mock.patch("subprocess.run") as m:
        built = build_image("autoexplore/llava", tmp_path)
        assert built is False        # 未构建
        m.assert_not_called()

def test_build_image_builds_when_missing(tmp_path):
    with mock.patch("scripts.docker_env.image_exists", return_value=False), \
         mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        built = build_image("autoexplore/llava", tmp_path)
        assert built is True
        called = m.call_args.args[0]
        assert called[:3] == ["docker", "build", "-t"]
        assert "autoexplore/llava" in called

def test_build_run_command_passes_gpus_and_mounts():
    cmd = build_run_command(
        image="autoexplore/llava", gpus="2,5",
        mounts=[("/host/run", "/work")], inner_cmd="python infer.py",
    )
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--gpus" in cmd
    gi = cmd.index("--gpus")
    assert cmd[gi + 1] == '"device=2,5"'
    assert "-v" in cmd
    vi = cmd.index("-v")
    assert cmd[vi + 1] == "/host/run:/work"
    # 镜像名后接 sh -c <inner_cmd>
    assert cmd[-4:] == ["autoexplore/llava", "sh", "-c", "python infer.py"]

def test_run_in_container_returns_exit_code(tmp_path):
    log = tmp_path / "run.log"
    with mock.patch("subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess([], 0, "", "")
        rc = run_in_container(
            image="img", gpus="0", mounts=[], inner_cmd="true", log_path=log,
        )
        assert rc == 0
        assert log.exists()

def test_check_runtime_raises_without_docker():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(DockerError, match="找不到 docker"):
            check_runtime()
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_docker_env.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.docker_env'`。

- [ ] **Step 3: 实现 docker_env.py**

```python
"""Docker 环境检查、镜像构建复用、容器内执行命令。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class DockerError(RuntimeError):
    """Docker 不可用或操作失败。"""


def check_runtime() -> None:
    """验证 docker 可用且支持 GPU,缺失则 fail fast。"""
    try:
        info = subprocess.run(
            ["docker", "info"], capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise DockerError("找不到 docker:本机未安装或不在 PATH") from e
    if info.returncode != 0:
        raise DockerError(f"docker 不可用: {info.stderr.strip()}")


def image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def build_image(tag: str, dockerfile_dir: Path) -> bool:
    """镜像已存在则跳过,返回 False;否则构建并返回 True。"""
    if image_exists(tag):
        return False
    proc = subprocess.run(
        ["docker", "build", "-t", tag, str(dockerfile_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise DockerError(f"镜像构建失败: {proc.stderr.strip()}")
    return True


def build_run_command(
    image: str, gpus: str, mounts: list[tuple[str, str]], inner_cmd: str
) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", f'"device={gpus}"']
    for host, container in mounts:
        cmd += ["-v", f"{host}:{container}"]
    cmd += [image, "sh", "-c", inner_cmd]
    return cmd


def run_in_container(
    image: str, gpus: str, mounts: list[tuple[str, str]],
    inner_cmd: str, log_path: Path,
) -> int:
    """跑容器,stdout+stderr 重定向到 log_path,返回退出码。"""
    cmd = build_run_command(image, gpus, mounts, inner_cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Docker 环境工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="检查 docker runtime")

    b = sub.add_parser("build", help="构建/复用镜像")
    b.add_argument("--tag", required=True)
    b.add_argument("--dockerfile-dir", type=Path, required=True)

    r = sub.add_parser("run", help="容器内执行命令")
    r.add_argument("--tag", required=True)
    r.add_argument("--gpus", default="")
    r.add_argument("--mount", action="append", default=[], help="host:container")
    r.add_argument("--log", type=Path, required=True)
    r.add_argument("--inner-cmd", required=True)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "check":
            check_runtime()
            print("docker runtime OK")
            return 0
        if args.cmd == "build":
            built = build_image(args.tag, args.dockerfile_dir)
            print("built" if built else "reused")
            return 0
        if args.cmd == "run":
            mounts = [tuple(m.split(":", 1)) for m in args.mount]
            rc = run_in_container(
                args.tag, args.gpus, mounts, args.inner_cmd, args.log
            )
            print(f"exit_code: {rc}")
            return rc
    except DockerError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_docker_env.py -v`
Expected: 7 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/docker_env.py tests/test_docker_env.py
git commit -m "feat(docker): runtime check, image reuse, GPU container exec"
```

---

## Task 6: run_inference.py — 容器内推理产出 predictions

职责:在容器内对 dataset 跑模型推理脚本,产出 `predictions.jsonl`(每行一条 JSON)。它是 docker_env 的薄封装 + 路径约定:把 run 目录挂进容器,调用模型自带或 Claude 自建的推理脚本,推理脚本的输出路径固定为容器内 `/work/predictions.jsonl`。本任务测推理命令拼装与产物路径约定,不真跑容器。

**Files:**
- Create: `scripts/run_inference.py`
- Test: 复用 `tests/test_docker_env.py` 的 mock 模式,新增 `tests/test_run_inference.py`

- [ ] **Step 1: 写失败测试**

`tests/test_run_inference.py`:

```python
from unittest import mock
import subprocess
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_run_inference.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.run_inference'`。

- [ ] **Step 3: 实现 run_inference.py**

```python
"""在容器内对 dataset 跑推理,产出 predictions.jsonl(每行一条 JSON)。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.docker_env import run_in_container

# run 目录挂载到容器内 /work;推理脚本须把输出写到这里
CONTAINER_WORK = "/work"
INNER_PREDICTIONS_PATH = "/work/predictions.jsonl"


def run_inference(
    image: str, gpus: str, run_dir: Path, infer_cmd: str, log_path: Path
) -> Path:
    """挂载 run_dir 到 /work,执行 infer_cmd;成功返回宿主侧 predictions 路径。

    infer_cmd 由 Claude 提供,约定把预测写到 INNER_PREDICTIONS_PATH。
    """
    rc = run_in_container(
        image=image,
        gpus=gpus,
        mounts=[(str(run_dir), CONTAINER_WORK)],
        inner_cmd=f"{infer_cmd} && test -f {INNER_PREDICTIONS_PATH}",
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(f"推理失败 (exit={rc}),详见 {log_path}")
    return run_dir / "predictions.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="容器内推理")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--infer-cmd", required=True)
    args = ap.parse_args(argv)
    try:
        preds = run_inference(
            args.tag, args.gpus, args.run_dir, args.infer_cmd, args.log
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"predictions: {preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_run_inference.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/run_inference.py tests/test_run_inference.py
git commit -m "feat(inference): run model inference in container, emit predictions"
```

---

## Task 7: 端到端 dry-run 契约测试

不碰真实 Docker:用玩具 dataset + 玩具 evaluate.py,把 stub 推理(直接写 predictions)接到真实的 progress / compute_metrics,验证整条**文件流转**与 schema 互相对得上(candidates.json → predictions.jsonl → metrics.json → results.tsv)。这是锁住脚本间接口的核心测试。

**Files:**
- Test: `tests/test_contract_dryrun.py`

- [ ] **Step 1: 写契约测试**

`tests/test_contract_dryrun.py`:

```python
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
```

- [ ] **Step 2: 运行确认通过**

Run: `uv run pytest tests/test_contract_dryrun.py -v`
Expected: 2 passed。

- [ ] **Step 3: 运行全部测试确认整体绿**

Run: `uv run pytest -v`
Expected: 全部 passed(gpu marker 测试此时尚未加入)。

- [ ] **Step 4: Commit**

```bash
git add tests/test_contract_dryrun.py
git commit -m "test(contract): lock phase-1 file-flow schema across scripts"
```

---

## Task 8: GPU 冒烟测试(默认 skip,服务器手动跑)

用极小公开镜像(`hello-world` 验证 docker,或一个 tiny python 镜像)验证 docker_env 真实路径,标 `@pytest.mark.gpu` 默认跳过。不依赖具体模型,只验证容器执行与日志重定向在真实环境工作。

**Files:**
- Test: `tests/test_smoke_gpu.py`

- [ ] **Step 1: 写冒烟测试**

`tests/test_smoke_gpu.py`:

```python
"""真实 Docker 冒烟测试。默认 skip,在 GPU 服务器上用 `pytest -m gpu` 跑。"""
import pytest
from scripts.docker_env import check_runtime, build_run_command, run_in_container

pytestmark = pytest.mark.gpu


def test_runtime_available():
    check_runtime()  # 不抛异常即通过


def test_container_echo_writes_log(tmp_path):
    log = tmp_path / "run.log"
    rc = run_in_container(
        image="python:3.12-slim", gpus="",
        mounts=[], inner_cmd="echo hello-autoexplore", log_path=log,
    )
    assert rc == 0
    assert "hello-autoexplore" in log.read_text()


def test_gpu_visible_in_container(tmp_path):
    """验证 --gpus 透传:容器内能看到 nvidia-smi。"""
    log = tmp_path / "gpu.log"
    rc = run_in_container(
        image="nvidia/cuda:12.4.0-base-ubuntu22.04", gpus="0",
        mounts=[], inner_cmd="nvidia-smi -L", log_path=log,
    )
    assert rc == 0
    assert "GPU 0" in log.read_text()
```

- [ ] **Step 2: 确认默认被跳过**

Run: `uv run pytest tests/test_smoke_gpu.py -v`
Expected: 3 deselected(因 `addopts = -m 'not gpu'`)。

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_gpu.py
git commit -m "test(smoke): add gpu-marked real-docker smoke tests"
```

---

## Task 9: 复现循环细则文档 references/reproduction-loop.md

把 spec 第 5 节细则落成 Claude 执行时引用的操作手册,包含每阶段对应的脚本命令。

**Files:**
- Create: `references/reproduction-loop.md`

- [ ] **Step 1: 写文档**

`references/reproduction-loop.md`:

````markdown
# 复现循环细则(单个候选模型)

对 `runs/<tag>/candidates.json` 中每个候选,按 priority 升序,在 `runs/<tag>/models/<name>/` 工作目录内执行。开始前读 `progress.json`,若 `stage` 为 `ready`/`crash` 则跳过,否则从记录的 stage 恢复。

## 阶段 A:准备环境

1. 选卡(每次进入都重查,因为别的进程可能占卡):
   ```bash
   GPUS=$(uv run scripts/gpu_select.py --count 1 --min-free-mib 20000)
   ```
   非零退出表示无空闲卡——决定等待或减小 --count / --min-free-mib。
2. 写一份 `Dockerfile`(参照模型仓库 README/requirements,由 Claude 生成)放进 `models/<name>/`。
3. 构建/复用镜像:
   ```bash
   uv run scripts/docker_env.py build --tag autoexplore/<name> --dockerfile-dir runs/<tag>/models/<name>
   ```
   输出 `reused` 表示已存在跳过(对齐"已构建过则忽略")。
4. 记录进度:
   ```bash
   uv run scripts/progress.py set --model-dir runs/<tag>/models/<name> \
     --model <name> --stage A --gpus "$GPUS" --image-tag autoexplore/<name>
   ```

## 阶段 B:下载

5. 在容器内拉安装包 + 模型权重,输出进 run.log:
   ```bash
   uv run scripts/docker_env.py run --tag autoexplore/<name> --gpus "$GPUS" \
     --mount "$(pwd)/runs/<tag>/models/<name>:/work" \
     --log runs/<tag>/models/<name>/run.log \
     --inner-cmd "<下载命令>"
   ```
6. 失败转阶段 D;成功 `progress.py set ... --stage B`。

## 阶段 C:验证推理

7. 有官方示例代码用示例;没有则 Claude 自建最小推理脚本,放进 `models/<name>/`。
8. 跑推理验证(对 dataset 产出 predictions.jsonl 到 /work/predictions.jsonl):
   ```bash
   uv run scripts/run_inference.py --tag autoexplore/<name> --gpus "$GPUS" \
     --run-dir runs/<tag>/models/<name> \
     --log runs/<tag>/models/<name>/run.log \
     --infer-cmd "<推理命令,输出写到 /work/predictions.jsonl>"
   ```
9. 成功 → `progress.py set ... --stage ready`,循环结束。失败 → 阶段 D。

## 阶段 D:有界重试

10. `retry_count += 1` 写进 progress.json。
11. 读 `tail -n 50 run.log` 的错误:
    - 简单可修(缺依赖/路径/CUDA 版本):改 Dockerfile 或命令,回阶段 A。
    - 根本性失败(架构不兼容/资源不够):放弃。
12. `retry_count >= 3` → 放弃,`progress.py set ... --stage crash`,在 results.tsv 记 crash:
    ```bash
    uv run scripts/progress.py result --run-dir runs/<tag> --model <name> \
      --primary-metric 0.0 --memory-gb 0.0 --status crash --description "<原因>"
    ```

## 成功后:推理 + 评测(步骤 8–9)

阶段 C 已产出 predictions.jsonl,计算指标:
```bash
uv run scripts/compute_metrics.py --evaluate-py runs/<tag>/evaluate.py \
  --predictions runs/<tag>/models/<name>/predictions.jsonl \
  --dataset runs/<tag>/dataset \
  --out runs/<tag>/models/<name>/metrics.json
```
记录结果:
```bash
uv run scripts/progress.py result --run-dir runs/<tag> --model <name> \
  --primary-metric <值> --memory-gb <值> --status ready --description "<复现说明>"
```

## 日志纪律

- 容器输出一律进 run.log,只在失败时 `tail -n 50 run.log`,绝不让全量日志进上下文。
- 重试上限 3 次(对齐 CLAUDE.md "3 次后停下重新评估")。
````

- [ ] **Step 2: Commit**

```bash
git add references/reproduction-loop.md
git commit -m "docs(loop): add reproduction-loop reference for the skill"
```

---

## Task 10: SKILL.md — 流程与决策编排

把 spec 第 4 节的 10 步编排成 Claude 执行手册,引用 reproduction-loop.md。

**Files:**
- Create: `SKILL.md`

- [ ] **Step 1: 写 SKILL.md**

`SKILL.md`:

````markdown
---
name: autoexplore-phase1
description: Use when reproducing open-source models for a research direction — drives the full phase-1 pipeline (clarify direction, build test set, design metrics, search/rank models, reproduce in Docker, infer, score, pick best).
---

# autoexplore 第一阶段:开源模型自动复现

输入:一段研究方向描述。产出:在该方向上复现并评测 ≤3 个开源模型,选出指标最高者。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡,按空闲度自动选卡。

## 流程(需求文档第一阶段步骤 1–10)

### 步骤 1–3:与用户确认(唯一的人工关卡)

1. **澄清方向**:与用户对话明确研究方向。商定一个 run-tag(如 `vqa-mar5`),创建 `runs/<tag>/`,写 `plan.md` 草稿。
2. **建立测试集**:按方向提议数据来源与规模,**人工确认后**落到 `runs/<tag>/dataset/`。此后不可变。
3. **设计指标**:写 `runs/<tag>/evaluate.py`,接口为
   `python evaluate.py --predictions <jsonl> --dataset <dir> --out <metrics.json>`,
   输出 `{"primary_metric": <float>, "metrics": {...}}`。**人工确认后冻结。**
   在 `plan.md` 里约定 predictions.jsonl 的每行 JSON 字段(供推理脚本与 evaluate.py 共同遵守)。

把步骤 1–3 的结论都写进 `plan.md`,得到用户确认后再进入步骤 4。之后尽量自主,不再逐步问人。

### 步骤 4–5:搜索与排序模型

4. **搜索**:在 Hugging Face、paperswithcode、arxiv 检索该方向可用开源模型。
5. **排序**:按官方披露效果自高到低排序,选 ≤3 个,写 `runs/<tag>/candidates.json`:
   ```json
   [{"name": "...", "repo": "...", "reported_score": 0.0, "priority": 1}]
   ```

### 步骤 6–9:逐个复现 + 推理 + 评测

按 priority 升序,对每个候选执行**复现循环**。完整细则见
[references/reproduction-loop.md](references/reproduction-loop.md):准备环境(选卡、构建/复用镜像)→ 下载 → 验证推理 → 有界重试(上限 3 次)→ 推理产出 predictions → compute_metrics 算分 → 写 results.tsv。

### 步骤 10:选最佳模型

读 `runs/<tag>/results.tsv`,在 `status=ready` 的模型里选 `primary_metric` 最高者,向用户报告。该模型是第二阶段(效果迭代)的候选。

## 关键纪律

- `dataset/` 与 `evaluate.py` 人工确认后**不可变**,保证模型间可比。
- 容器输出进 `run.log`,只在失败时 tail,不污染上下文。
- 每模型重试上限 3 次,失败标 `crash` 但不阻塞其他模型。
- 复现串行,一次一个(第一版不并行、不训练)。
- 中断可恢复:复现循环开始读 `progress.json`,跳过已 ready/crash 的模型。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/gpu_select.py --count N --min-free-mib M` | 选卡 → 打印 CUDA_VISIBLE_DEVICES |
| `scripts/docker_env.py {check,build,run}` | Docker 检查 / 镜像构建复用 / 容器执行 |
| `scripts/run_inference.py` | 容器内推理 → predictions.jsonl |
| `scripts/compute_metrics.py` | 调 evaluate.py → metrics.json |
| `scripts/progress.py {set,done,result}` | 进度持久化 / 终态查询 / 结果汇总 |
````

- [ ] **Step 2: 校验 skill frontmatter 与引用文件存在**

Run: `test -f references/reproduction-loop.md && head -3 SKILL.md`
Expected: 打印 frontmatter 前三行(`---` / `name:` / `description:`),且引用文件存在。

- [ ] **Step 3: 运行全部非 gpu 测试做最终回归**

Run: `uv run pytest -v`
Expected: 全部 passed,gpu 测试 deselected。

- [ ] **Step 4: Commit**

```bash
git add SKILL.md
git commit -m "feat(skill): add phase-1 reproduction orchestration skill"
```

---

## 完成标准

- [ ] 5 个脚本各有单元测试,全部通过(`uv run pytest`)。
- [ ] 契约测试锁住脚本间文件流转 schema。
- [ ] GPU 冒烟测试存在并默认 skip,可在服务器 `uv run pytest -m gpu` 跑。
- [ ] SKILL.md + reproduction-loop.md 覆盖步骤 1–10。
- [ ] 全部提交遵循 Conventional Commits(GIT_CONVENTIONS.md)。
