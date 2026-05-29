# 分层模型 Web 部署 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `runs/design-layer-may26` 的 qwen-image-layered / layerd 两个分层模型做一个 Gradio web 部署：选模型、传图、得到叠加在一张画布上、可自由拖拽的 RGBA 分层结果。

**Architecture:** 前端 Gradio（host `.venv`，:7860）+ 每模型一个常驻后端容器（Python stdlib `http.server`，模型懒加载进 GPU、空闲 600s 卸载，复用 `runs/.../models/<m>/infer.py` 的加载/推理函数）。qwen 推理后强制丢空层，按空闲显存自动选 bf16/int8。拖拽/显隐/zorder/透明度全在前端 JS。

**Tech Stack:** Python 3.14 (host) / 3.11 (容器 conda)、Gradio、stdlib http.server、requests、numpy、Pillow、Docker (nvidia runtime)、既有镜像 `autoexplore/qwen-image-layered`、`autoexplore/layerd`。

**约定：** 全部代码在 `web/`。测试在 `web/tests/`，用 `.venv/bin/pytest web/tests` 运行；GPU 冒烟测试用 `pytestmark = pytest.mark.gpu`（`pyproject.toml` 已注册该 marker 并默认 `-m 'not gpu'` 跳过）。分支 `feature/web-layered-deploy`。提交用 Conventional Commits，scope=`web`，commit 末尾加 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。

---

## 文件结构

```
web/
  __init__.py
  config.yaml                # 端口 / idle_ttl / 各模型默认参数 / drop-empty 阈值
  common/
    __init__.py
    imaging.py               # encode_png_b64 / decode_b64_png （前后端共享）
    postprocess.py           # coverage / alpha_bbox / drop_empty （phase2 同语义）
    config.py                # DEFAULTS / load_config / merge_params
  backend/
    __init__.py
    runners.py               # choose_qwen_dtype + Runner 协议 + Qwen/Layerd Runner + assemble_response
    server.py                # ModelManager + http.server handler + main
  frontend/
    __init__.py
    client.py                # predict / health （HTTP 客户端）
    ui.py                    # canvas_html + DRAG_JS （画布/面板 HTML 与拖拽 JS）
    app.py                   # Gradio Blocks 装配
  scripts/
    run_backend_qwen.sh
    run_backend_layerd.sh
    run_frontend.sh
    stop_all.sh
  tests/
    __init__.py
    test_imaging.py
    test_postprocess.py
    test_config.py
    test_runners.py
    test_assemble.py
    test_manager.py
    test_client.py
    test_ui.py
    test_backend_smoke.py    # gpu-marked, 默认 skip
  README.md
```

接口契约（跨任务一致，先在此锁定）：
- `imaging.encode_png_b64(img: PIL.Image) -> str`，`imaging.decode_b64_png(s: str) -> PIL.Image`
- `postprocess.coverage(rgba) -> float`；`postprocess.alpha_bbox(rgba) -> list[int] | None`（`[x,y,w,h]`）；`postprocess.drop_empty(layers, thresh=0.0) -> tuple[list[Image], list[float]]`
- `config.load_config(path=None) -> dict`；`config.merge_params(model, user_params, cfg) -> dict`
- `runners.choose_qwen_dtype(free_mib, bf16_min=40000) -> str`（`"bf16"`/`"int8"`）
- `runners.Runner`：属性 `name`、`dtype`、`is_loaded`；方法 `load()`、`infer(image, params) -> list[Image]`、`unload()`
- `runners.assemble_response(model, dtype, infer_ms, raw_layers, thresh) -> dict`
- `server.ModelManager(runner_factory, idle_ttl=600.0)`：`get(now) -> Runner`、`maybe_unload(now) -> bool`、属性 `loaded`
- `client.predict(base_url, image, params, timeout=600) -> dict`；`client.health(base_url, timeout=5) -> dict`
- `ui.canvas_html(response: dict) -> str`；`ui.DRAG_JS: str`

后端 `/predict` 响应形如：
```json
{"model":"qwen-image-layered","dtype":"int8","infer_ms":12345,
 "canvas":{"w":640,"h":853},
 "layers":[{"z_order":0,"png_b64":"...","coverage":0.42,"bbox":[10,20,300,400]}],
 "dropped":3}
```

---

### Task 0: 脚手架与依赖

**Files:**
- Create: `web/__init__.py`, `web/common/__init__.py`, `web/backend/__init__.py`, `web/frontend/__init__.py`, `web/tests/__init__.py`
- Create: `web/config.yaml`

- [ ] **Step 1: 建包目录与空 `__init__.py`**

```bash
cd /dev_share/zbchu2/autoexplore
mkdir -p web/common web/backend web/frontend web/scripts web/tests
touch web/__init__.py web/common/__init__.py web/backend/__init__.py web/frontend/__init__.py web/tests/__init__.py
```

- [ ] **Step 2: 写 `web/config.yaml`**

```yaml
# Web 部署配置。脚本与 frontend/backend 都读这个文件。
ports:
  frontend: 7860
  qwen-image-layered: 8801
  layerd: 8802
idle_ttl: 600          # 秒；模型空闲超过此值从显存卸载
models:
  qwen-image-layered:
    layers: 8
    steps: 30
    resolution: 640
    drop_empty_thresh: 0.0     # 覆盖度 <= thresh 的层丢弃（0 = 仅丢真空层）
    bf16_min_free_mib: 40000   # 空闲显存 >= 此值用 bf16，否则 int8
  layerd:
    max_iterations: 6
    drop_empty_thresh: 0.0
```

- [ ] **Step 3: 安装前端依赖 gradio（host .venv）**

Run（清华镜像，见网络偏好记忆；numpy/PIL/requests/yaml/pytest 已具备）：
```bash
cd /dev_share/zbchu2/autoexplore
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gradio
.venv/bin/python -c "import gradio; print('gradio', gradio.__version__)"
```
Expected: 打印出 gradio 版本号，无报错。

- [ ] **Step 4: Commit**

```bash
git add web/__init__.py web/common/__init__.py web/backend/__init__.py web/frontend/__init__.py web/tests/__init__.py web/config.yaml
git commit -m "chore(web): scaffold web/ package layout and config"
```

---

### Task 1: `common/imaging.py` — PNG ↔ base64

**Files:**
- Create: `web/common/imaging.py`
- Test: `web/tests/test_imaging.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_imaging.py
from PIL import Image
from web.common.imaging import encode_png_b64, decode_b64_png


def test_roundtrip_preserves_rgba_pixels():
    img = Image.new("RGBA", (4, 3), (10, 20, 30, 40))
    img.putpixel((1, 1), (200, 100, 50, 255))
    s = encode_png_b64(img)
    assert isinstance(s, str) and len(s) > 0
    out = decode_b64_png(s)
    assert out.mode == "RGBA"
    assert out.size == (4, 3)
    assert out.getpixel((1, 1)) == (200, 100, 50, 255)


def test_encode_is_pure_base64_no_data_uri_prefix():
    s = encode_png_b64(Image.new("RGBA", (2, 2)))
    assert not s.startswith("data:")  # 纯 base64，前端再自己拼 data: 前缀
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_imaging.py -v`
Expected: FAIL（`ModuleNotFoundError: web.common.imaging`）

- [ ] **Step 3: 实现**

```python
# web/common/imaging.py
"""PNG <-> base64 编解码，前后端共享。返回的是纯 base64（不含 data: 前缀）。"""
from __future__ import annotations

import base64
import io

from PIL import Image


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_b64_png(s: str) -> Image.Image:
    if s.startswith("data:"):  # 容忍带 data URI 前缀的输入
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGBA")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_imaging.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add web/common/imaging.py web/tests/test_imaging.py
git commit -m "feat(web): add png/base64 imaging helpers"
```

---

### Task 2: `common/postprocess.py` — 丢空层（与 phase2 同语义）

**Files:**
- Create: `web/common/postprocess.py`
- Test: `web/tests/test_postprocess.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_postprocess.py
from PIL import Image
from web.common.postprocess import coverage, alpha_bbox, drop_empty


def _solid(alpha, size=(10, 10)):
    return Image.new("RGBA", size, (255, 0, 0, alpha))


def _empty(size=(10, 10)):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def test_coverage_full_and_empty():
    assert coverage(_solid(255)) == 1.0
    assert coverage(_empty()) == 0.0


def test_coverage_threshold_is_alpha_ge_half():
    # alpha=100 (<128) -> 不计入覆盖；alpha=200 (>=128) -> 计入
    assert coverage(_solid(100)) == 0.0
    assert coverage(_solid(200)) == 1.0


def test_alpha_bbox_tight():
    img = _empty((10, 10))
    img.putpixel((3, 4), (255, 255, 255, 255))
    img.putpixel((6, 7), (255, 255, 255, 255))
    assert alpha_bbox(img) == [3, 4, 4, 4]  # x,y,w,h 覆盖 (3,4)..(6,7)


def test_alpha_bbox_none_when_empty():
    assert alpha_bbox(_empty()) is None


def test_drop_empty_removes_only_zero_coverage():
    layers = [_solid(255), _empty(), _solid(255)]
    kept, covs = drop_empty(layers, thresh=0.0)
    assert len(kept) == 2
    assert covs == [1.0, 1.0]


def test_drop_empty_keeps_highest_when_all_empty():
    a, b = _empty(), _solid(255)
    # b 覆盖度更高但 thresh=1.0 把它也判为"空"——仍须至少保留覆盖最高的一层
    kept, covs = drop_empty([a, b], thresh=1.0)
    assert kept == [b]
    assert covs == [1.0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_postprocess.py -v`
Expected: FAIL（`ModuleNotFoundError: web.common.postprocess`）

- [ ] **Step 3: 实现**

```python
# web/common/postprocess.py
"""丢弃近空预测层。语义与 runs/design-layer-may26/phase2/postprocess_drop_empty.py 一致：
覆盖度 = 二值化 alpha(>=0.5) 的像素占比；丢掉覆盖度 <= thresh 的层，但至少保留覆盖度最高的一层。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def coverage(rgba: Image.Image) -> float:
    a = np.asarray(rgba.convert("RGBA").split()[-1], dtype=np.float32) / 255.0
    return float((a >= 0.5).mean())


def alpha_bbox(rgba: Image.Image) -> list[int] | None:
    a = np.asarray(rgba.convert("RGBA").split()[-1])
    ys, xs = np.where(a > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def drop_empty(
    layers: list[Image.Image], thresh: float = 0.0
) -> tuple[list[Image.Image], list[float]]:
    scored = [(lyr, coverage(lyr)) for lyr in layers]
    kept = [(lyr, c) for lyr, c in scored if c > thresh]
    if not kept:  # 永不返回 0 层
        kept = [max(scored, key=lambda lc: lc[1])]
    return [lyr for lyr, _ in kept], [c for _, c in kept]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_postprocess.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add web/common/postprocess.py web/tests/test_postprocess.py
git commit -m "feat(web): add drop-empty postprocess matching phase2 semantics"
```

---

### Task 3: `common/config.py` — 配置与参数合并

**Files:**
- Create: `web/common/config.py`
- Test: `web/tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_config.py
from web.common.config import load_config, merge_params, DEFAULTS


def test_load_config_returns_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg["idle_ttl"] == DEFAULTS["idle_ttl"]
    assert cfg["ports"]["frontend"] == 7860


def test_load_config_merges_user_over_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("idle_ttl: 30\nports:\n  frontend: 9000\n")
    cfg = load_config(p)
    assert cfg["idle_ttl"] == 30          # 用户覆盖
    assert cfg["ports"]["frontend"] == 9000
    assert cfg["ports"]["layerd"] == 8802  # 未覆盖项保留默认


def test_merge_params_fills_qwen_defaults():
    cfg = load_config(None)
    p = merge_params("qwen-image-layered", {"steps": 50}, cfg)
    assert p["steps"] == 50      # 用户值
    assert p["layers"] == 8      # 默认值
    assert p["resolution"] == 640


def test_merge_params_ignores_unknown_keys():
    cfg = load_config(None)
    p = merge_params("layerd", {"steps": 99, "max_iterations": 4}, cfg)
    assert p["max_iterations"] == 4
    assert "steps" not in p      # layerd 不认 steps
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: web.common.config`）

- [ ] **Step 3: 实现**

```python
# web/common/config.py
"""读 web/config.yaml 并与内置默认合并；按模型裁剪/补全推理参数。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "ports": {"frontend": 7860, "qwen-image-layered": 8801, "layerd": 8802},
    "idle_ttl": 600,
    "models": {
        "qwen-image-layered": {
            "layers": 8, "steps": 30, "resolution": 640,
            "drop_empty_thresh": 0.0, "bf16_min_free_mib": 40000,
        },
        "layerd": {"max_iterations": 6, "drop_empty_thresh": 0.0},
    },
}

# 每个模型推理时实际接受的参数键（merge_params 只保留这些）
_INFER_KEYS = {
    "qwen-image-layered": ("layers", "steps", "resolution"),
    "layerd": ("max_iterations",),
}

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None) -> dict:
    p = Path(path) if path is not None else _DEFAULT_PATH
    user = {}
    if p.exists():
        user = yaml.safe_load(p.read_text()) or {}
    return _deep_merge(DEFAULTS, user)


def merge_params(model: str, user_params: dict, cfg: dict) -> dict:
    model_cfg = cfg["models"][model]
    out = {}
    for k in _INFER_KEYS[model]:
        out[k] = (user_params or {}).get(k, model_cfg[k])
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_config.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add web/common/config.py web/tests/test_config.py
git commit -m "feat(web): add config loader and per-model param merge"
```

---

### Task 4: `backend/runners.py` — dtype 选择 + Runner（GPU 部分懒导入）

**Files:**
- Create: `web/backend/runners.py`
- Test: `web/tests/test_runners.py`

注意：`torch`/`diffusers`/`layerd` 只在 `Runner.load()` 内部 import（容器内才有），模块顶层只用 stdlib，保证 host 上能 import 本模块测 `choose_qwen_dtype`。

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_runners.py
from web.backend.runners import choose_qwen_dtype, build_runner


def test_choose_qwen_dtype_threshold():
    assert choose_qwen_dtype(40000) == "bf16"   # 恰好达到阈值
    assert choose_qwen_dtype(45000) == "bf16"
    assert choose_qwen_dtype(39999) == "int8"
    assert choose_qwen_dtype(20000) == "int8"


def test_choose_qwen_dtype_custom_min():
    assert choose_qwen_dtype(30000, bf16_min=25000) == "bf16"


def test_build_runner_unknown_model_raises():
    import pytest
    with pytest.raises(ValueError):
        build_runner("nope", free_mib=10000)


def test_build_runner_sets_name_and_dtype_without_loading():
    # build_runner 不应触发 torch import / 模型加载
    q = build_runner("qwen-image-layered", free_mib=20000)
    assert q.name == "qwen-image-layered"
    assert q.dtype == "int8"
    assert q.is_loaded is False
    ld = build_runner("layerd", free_mib=8000)
    assert ld.name == "layerd"
    assert ld.dtype == "n/a"
    assert ld.is_loaded is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_runners.py -v`
Expected: FAIL（`ModuleNotFoundError: web.backend.runners`）

- [ ] **Step 3: 实现**

```python
# web/backend/runners.py
"""模型适配层：把已有 infer.py 的加载/推理逻辑包成统一 Runner，供 server 懒加载调用。
torch / diffusers / layerd 仅在 load() 内 import（只有容器里才装了）。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PIL import Image

from web.common.imaging import encode_png_b64
from web.common.postprocess import alpha_bbox, drop_empty

# 容器内 RUN_DIR 挂载点（run_backend_*.sh 用 -v RUN_DIR:/work）
RUN_DIR = Path(os.environ.get("RUN_DIR_IN_CONTAINER", "/work"))

QWEN_DIR = RUN_DIR / "models" / "qwen-image-layered"
QWEN_Q8_DIR = RUN_DIR / "phase2" / "inference_tuning" / "exp_layers16"
LAYERD_DIR = RUN_DIR / "models" / "layerd"


def choose_qwen_dtype(free_mib: int, bf16_min: int = 40000) -> str:
    return "bf16" if free_mib >= bf16_min else "int8"


def _import_from(dir_path: Path, module_name: str):
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))
    return __import__(module_name)


class QwenRunner:
    name = "qwen-image-layered"

    def __init__(self, dtype: str, model_path: str | None = None):
        self.dtype = dtype
        self._model_path = model_path or os.environ.get(
            "MODEL_PATH", "/root/.cache/modelscope/hub/models/Qwen/Qwen-Image-Layered"
        )
        self._pipe = None
        self._infer_one = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        # int8 走 infer_q8.load_pipeline；bf16 走基线 infer.load_pipeline。
        base = _import_from(QWEN_DIR, "infer")
        self._infer_one = base.infer_one  # 两条路径推理逻辑一致，复用基线 infer_one
        if self.dtype == "int8":
            q8 = _import_from(QWEN_Q8_DIR, "infer_q8")
            self._pipe = q8.load_pipeline(self._model_path)
        else:
            self._pipe = base.load_pipeline(self._model_path)

    def infer(self, image: Image.Image, params: dict) -> list[Image.Image]:
        return self._infer_one(
            self._pipe, image.convert("RGB"),
            params["resolution"], params["layers"], params["steps"],
        )

    def unload(self) -> None:
        import torch
        self._pipe = None
        self._infer_one = None
        torch.cuda.empty_cache()


class LayerdRunner:
    name = "layerd"
    dtype = "n/a"

    def __init__(self, matting_card: str | None = None):
        self._matting_card = matting_card or os.environ.get(
            "MATTING_CARD", "cyberagent/layerd-birefnet"
        )
        self._inst = None

    @property
    def is_loaded(self) -> bool:
        return self._inst is not None

    def load(self) -> None:
        base = _import_from(LAYERD_DIR, "infer")
        self._inst = base.load_layerd(self._matting_card, os.environ.get("HF_CACHE", "/cache/hf"))

    def infer(self, image: Image.Image, params: dict) -> list[Image.Image]:
        return self._inst.decompose(image.convert("RGBA"), max_iterations=params["max_iterations"])

    def unload(self) -> None:
        import torch
        self._inst = None
        torch.cuda.empty_cache()


def build_runner(model: str, free_mib: int, bf16_min: int = 40000):
    if model == "qwen-image-layered":
        return QwenRunner(dtype=choose_qwen_dtype(free_mib, bf16_min))
    if model == "layerd":
        return LayerdRunner()
    raise ValueError(f"unknown model: {model}")


def assemble_response(
    model: str, dtype: str, infer_ms: int,
    raw_layers: list[Image.Image], thresh: float,
) -> dict:
    kept, covs = drop_empty(raw_layers, thresh)
    layers = [
        {"z_order": z, "png_b64": encode_png_b64(lyr),
         "coverage": cov, "bbox": alpha_bbox(lyr)}
        for z, (lyr, cov) in enumerate(zip(kept, covs))
    ]
    return {
        "model": model, "dtype": dtype, "infer_ms": infer_ms,
        "canvas": {"w": kept[0].width, "h": kept[0].height},
        "layers": layers, "dropped": len(raw_layers) - len(kept),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_runners.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add web/backend/runners.py web/tests/test_runners.py
git commit -m "feat(web): add qwen/layerd runners with lazy gpu imports"
```

---

### Task 5: `assemble_response` 的 CPU 测试

**Files:**
- Test: `web/tests/test_assemble.py`（`assemble_response` 已在 Task 4 实现，这里补纯 CPU 测试）

- [ ] **Step 1: 写测试**

```python
# web/tests/test_assemble.py
from PIL import Image
from web.backend.runners import assemble_response
from web.common.imaging import decode_b64_png


def _solid(alpha, size=(8, 6)):
    return Image.new("RGBA", size, (10, 20, 30, alpha))


def test_assemble_drops_empty_and_encodes_layers():
    raw = [_solid(255), Image.new("RGBA", (8, 6), (0, 0, 0, 0)), _solid(255)]
    resp = assemble_response("qwen-image-layered", "int8", 1234, raw, thresh=0.0)
    assert resp["model"] == "qwen-image-layered"
    assert resp["dtype"] == "int8"
    assert resp["infer_ms"] == 1234
    assert resp["dropped"] == 1
    assert resp["canvas"] == {"w": 8, "h": 6}
    assert len(resp["layers"]) == 2
    # z_order 连续重排
    assert [l["z_order"] for l in resp["layers"]] == [0, 1]
    # png_b64 能解码回 RGBA 且尺寸一致
    img0 = decode_b64_png(resp["layers"][0]["png_b64"])
    assert img0.size == (8, 6) and img0.mode == "RGBA"
    assert resp["layers"][0]["bbox"] == [0, 0, 8, 6]
```

- [ ] **Step 2: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_assemble.py -v`
Expected: PASS（1 passed）

- [ ] **Step 3: Commit**

```bash
git add web/tests/test_assemble.py
git commit -m "test(web): cover assemble_response on cpu"
```

---

### Task 6: `backend/server.py` — ModelManager（懒加载/TTL/锁）

**Files:**
- Create: `web/backend/server.py`
- Test: `web/tests/test_manager.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_manager.py
from web.backend.server import ModelManager


class FakeRunner:
    def __init__(self):
        self.loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    def load(self):
        self.loaded = True
        self.load_calls += 1

    def unload(self):
        self.loaded = False
        self.unload_calls += 1


def test_get_loads_once_and_reuses():
    r = FakeRunner()
    mgr = ModelManager(lambda: r, idle_ttl=100)
    a = mgr.get(now=0.0)
    b = mgr.get(now=1.0)
    assert a is b is r
    assert r.load_calls == 1
    assert mgr.loaded is True


def test_maybe_unload_only_after_ttl():
    r = FakeRunner()
    mgr = ModelManager(lambda: r, idle_ttl=100)
    mgr.get(now=0.0)
    assert mgr.maybe_unload(now=50.0) is False   # 未超时
    assert mgr.loaded is True
    assert mgr.maybe_unload(now=200.0) is True    # 超时卸载
    assert mgr.loaded is False
    assert r.unload_calls == 1


def test_get_reloads_after_unload():
    r = FakeRunner()
    mgr = ModelManager(lambda: r, idle_ttl=100)
    mgr.get(now=0.0)
    mgr.maybe_unload(now=200.0)
    mgr.get(now=300.0)
    assert r.load_calls == 2
    assert mgr.loaded is True


def test_maybe_unload_noop_when_not_loaded():
    mgr = ModelManager(lambda: FakeRunner(), idle_ttl=100)
    assert mgr.maybe_unload(now=999.0) is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_manager.py -v`
Expected: FAIL（`ModuleNotFoundError: web.backend.server`，或 `cannot import name ModelManager`）

- [ ] **Step 3: 实现 server.py**

```python
# web/backend/server.py
"""单模型后端：stdlib http.server 暴露 /health 与 /predict；模型懒加载进 GPU、空闲 TTL 卸载。
启动时通过环境变量配置：MODEL（qwen-image-layered|layerd）、PORT、FREE_MIB、IDLE_TTL。"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from web.backend.runners import build_runner, assemble_response
from web.common.config import load_config, merge_params
from web.common.imaging import decode_b64_png


class ModelManager:
    def __init__(self, runner_factory, idle_ttl: float = 600.0):
        self._factory = runner_factory
        self._runner = None
        self._last_used = None
        self._lock = threading.Lock()
        self.idle_ttl = idle_ttl

    @property
    def loaded(self) -> bool:
        return self._runner is not None

    def get(self, now: float):
        with self._lock:
            if self._runner is None:
                r = self._factory()
                r.load()
                self._runner = r
            self._last_used = now
            return self._runner

    def maybe_unload(self, now: float) -> bool:
        with self._lock:
            if (self._runner is not None and self._last_used is not None
                    and now - self._last_used > self.idle_ttl):
                self._runner.unload()
                self._runner = None
                return True
            return False


def _make_handler(model: str, manager: ModelManager, cfg: dict, dtype_hint: str):
    thresh = cfg["models"][model]["drop_empty_thresh"]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # 静音默认 stderr 噪声
            pass

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"loaded": manager.loaded, "model": model,
                                 "dtype": dtype_hint,
                                 "free_mib": int(os.environ.get("FREE_MIB", "0"))})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                image = decode_b64_png(req["image_b64"])
                params = merge_params(model, req.get("params", {}), cfg)
                runner = manager.get(now=time.time())
                t0 = time.time()
                raw = runner.infer(image, params)
                infer_ms = int((time.time() - t0) * 1000)
                resp = assemble_response(model, runner.dtype, infer_ms, raw, thresh)
                self._send(200, resp)
            except Exception as e:  # 兜底：返回 JSON 错误而非崩连接
                hint = ""
                if "out of memory" in str(e).lower():
                    hint = "GPU 显存不足：试试更小的 layers，或让 qwen 走 int8。"
                self._send(500, {"error": repr(e), "hint": hint})

    return Handler


def _idle_loop(manager: ModelManager, stop: threading.Event):
    while not stop.wait(30):
        manager.maybe_unload(now=time.time())


def main() -> int:
    model = os.environ["MODEL"]
    cfg = load_config(os.environ.get("CONFIG_PATH"))
    port = int(os.environ.get("PORT", cfg["ports"][model]))
    idle_ttl = float(os.environ.get("IDLE_TTL", cfg["idle_ttl"]))
    free_mib = int(os.environ.get("FREE_MIB", "0"))
    bf16_min = cfg["models"].get(model, {}).get("bf16_min_free_mib", 40000)

    def factory():
        return build_runner(model, free_mib=free_mib, bf16_min=bf16_min)

    # 预先算出 dtype 供 /health 展示（不触发加载）
    dtype_hint = factory().dtype
    manager = ModelManager(factory, idle_ttl=idle_ttl)

    stop = threading.Event()
    threading.Thread(target=_idle_loop, args=(manager, stop), daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(model, manager, cfg, dtype_hint))
    print(f"[backend] model={model} port={port} idle_ttl={idle_ttl} dtype={dtype_hint} free_mib={free_mib}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_manager.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add web/backend/server.py web/tests/test_manager.py
git commit -m "feat(web): add backend http server with lazy load and idle ttl"
```

---

### Task 7: `frontend/client.py` — 后端 HTTP 客户端

**Files:**
- Create: `web/frontend/client.py`
- Test: `web/tests/test_client.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_client.py
from unittest import mock

from PIL import Image
from web.frontend import client


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_predict_posts_image_and_params():
    img = Image.new("RGBA", (4, 4), (1, 2, 3, 255))
    with mock.patch("web.frontend.client.requests.post",
                    return_value=_Resp(200, {"layers": [], "dropped": 0})) as p:
        out = client.predict("http://x:8801", img, {"steps": 30}, timeout=123)
    assert out == {"layers": [], "dropped": 0}
    url, = p.call_args.args
    assert url == "http://x:8801/predict"
    sent = p.call_args.kwargs["json"]
    assert sent["params"] == {"steps": 30}
    assert isinstance(sent["image_b64"], str) and len(sent["image_b64"]) > 0
    assert p.call_args.kwargs["timeout"] == 123


def test_predict_raises_on_http_error():
    img = Image.new("RGBA", (2, 2))
    with mock.patch("web.frontend.client.requests.post",
                    return_value=_Resp(500, {"error": "boom"})):
        import pytest
        with pytest.raises(RuntimeError):
            client.predict("http://x:8801", img, {})


def test_health_returns_json():
    with mock.patch("web.frontend.client.requests.get",
                    return_value=_Resp(200, {"loaded": False, "model": "layerd"})):
        out = client.health("http://x:8802")
    assert out["model"] == "layerd"


def test_health_returns_offline_marker_on_exception():
    with mock.patch("web.frontend.client.requests.get", side_effect=OSError("refused")):
        out = client.health("http://x:8802")
    assert out["loaded"] is False
    assert out.get("online") is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_client.py -v`
Expected: FAIL（`ModuleNotFoundError: web.frontend.client`）

- [ ] **Step 3: 实现**

```python
# web/frontend/client.py
"""前端 -> 后端 HTTP 客户端。"""
from __future__ import annotations

import requests
from PIL import Image

from web.common.imaging import encode_png_b64


def predict(base_url: str, image: Image.Image, params: dict, timeout: float = 600) -> dict:
    r = requests.post(
        f"{base_url}/predict",
        json={"image_b64": encode_png_b64(image), "params": params or {}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def health(base_url: str, timeout: float = 5) -> dict:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        r.raise_for_status()
        out = r.json()
        out["online"] = True
        return out
    except Exception:
        return {"loaded": False, "online": False}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_client.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add web/frontend/client.py web/tests/test_client.py
git commit -m "feat(web): add backend http client with health probe"
```

---

### Task 8: `frontend/ui.py` — 叠加画布 + 图层面板 HTML 与拖拽 JS

**Files:**
- Create: `web/frontend/ui.py`
- Test: `web/tests/test_ui.py`

- [ ] **Step 1: 写失败测试**

```python
# web/tests/test_ui.py
from web.frontend.ui import canvas_html, DRAG_JS


def _resp(n):
    return {
        "model": "qwen-image-layered", "dtype": "int8", "dropped": 2,
        "canvas": {"w": 100, "h": 80},
        "layers": [
            {"z_order": i, "png_b64": f"AAA{i}", "coverage": 0.5, "bbox": [0, 0, 100, 80]}
            for i in range(n)
        ],
    }


def test_canvas_html_one_img_per_layer():
    html = canvas_html(_resp(3))
    assert html.count("data:image/png;base64,") == 3
    assert "AAA0" in html and "AAA2" in html


def test_canvas_html_sets_canvas_size_and_zindex():
    html = canvas_html(_resp(2))
    assert "width:100px" in html.replace(" ", "")  # 容忍空格
    assert "z-index:0" in html.replace(" ", "")
    assert "z-index:1" in html.replace(" ", "")


def test_canvas_html_has_layer_panel_rows_and_controls():
    html = canvas_html(_resp(2))
    # 面板里每层一行：显隐、上移/下移、透明度、下载
    assert html.count('class="layer-row"') == 2
    assert "layer-vis" in html       # 显隐控件 class
    assert "layer-up" in html and "layer-down" in html
    assert "layer-opacity" in html
    assert "download" in html.lower()


def test_canvas_html_reports_dropped_and_dtype():
    html = canvas_html(_resp(1))
    assert "dropped" in html.lower() or "丢" in html
    assert "int8" in html


def test_drag_js_defines_init_and_observer():
    assert "function initLayeredCanvas" in DRAG_JS
    assert "MutationObserver" in DRAG_JS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest web/tests/test_ui.py -v`
Expected: FAIL（`ModuleNotFoundError: web.frontend.ui`）

- [ ] **Step 3: 实现**

```python
# web/frontend/ui.py
"""把后端响应渲染成「叠加画布 + 图层面板」HTML；拖拽/显隐/zorder/透明度全靠 DRAG_JS（前端 JS）。
DRAG_JS 通过 gr.Blocks(head=...) 注入；MutationObserver 监听新出现的 .layered-canvas 自动挂载。"""
from __future__ import annotations

import html as _html


def canvas_html(response: dict) -> str:
    w = response["canvas"]["w"]
    h = response["canvas"]["h"]
    layers = response["layers"]

    imgs = []
    rows = []
    for l in layers:
        z = l["z_order"]
        uri = "data:image/png;base64," + l["png_b64"]
        imgs.append(
            f'<img class="layer-img" data-z="{z}" draggable="false" '
            f'src="{uri}" style="position:absolute; left:0; top:0; '
            f'z-index:{z}; transform:translate(0px,0px); opacity:1;" />'
        )
        # 面板行：上=顶层，所以逆序展示
        rows.append(
            f'<div class="layer-row" data-z="{z}">'
            f'<input type="checkbox" class="layer-vis" data-z="{z}" checked /> '
            f'<span class="layer-name">L{z}</span> '
            f'<button class="layer-up" data-z="{z}">↑</button>'
            f'<button class="layer-down" data-z="{z}">↓</button>'
            f'<input type="range" class="layer-opacity" data-z="{z}" '
            f'min="0" max="100" value="100" /> '
            f'<a class="layer-dl" download="L{z}.png" '
            f'href="data:image/png;base64,{l["png_b64"]}">下载</a>'
            f'</div>'
        )

    info = (f'<div class="layer-info">模型 {_html.escape(response["model"])} · '
            f'dtype {_html.escape(str(response.get("dtype", "")))} · '
            f'丢空层 dropped={response.get("dropped", 0)} · '
            f'保留 {len(layers)} 层</div>')

    canvas = (
        f'<div class="layered-canvas" style="position:relative; '
        f'width:{w}px; height:{h}px; border:1px solid #ccc; '
        f'background:repeating-conic-gradient(#eee 0% 25%, #fff 0% 50%) 50%/16px 16px;">'
        + "".join(imgs) + "</div>"
    )
    panel = '<div class="layer-panel">' + "".join(reversed(rows)) + "</div>"

    return (
        '<div class="layered-wrap" style="display:flex; gap:16px; align-items:flex-start;">'
        + f'<div>{canvas}</div>'
        + f'<div style="min-width:240px;">{info}{panel}</div>'
        + '</div>'
    )


DRAG_JS = r"""
<script>
function initLayeredCanvas(root) {
  if (!root || root.dataset.lcInit) return;
  root.dataset.lcInit = "1";
  // 拖拽每个 .layer-img（更新 transform translate）
  root.querySelectorAll(".layer-img").forEach(function (img) {
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    function parse() {
      const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(img.style.transform);
      return m ? [parseFloat(m[1]), parseFloat(m[2])] : [0, 0];
    }
    img.addEventListener("pointerdown", function (e) {
      dragging = true; [ox, oy] = parse(); sx = e.clientX; sy = e.clientY;
      img.setPointerCapture(e.pointerId); e.preventDefault();
    });
    img.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      img.style.transform = "translate(" + (ox + e.clientX - sx) + "px," + (oy + e.clientY - sy) + "px)";
    });
    img.addEventListener("pointerup", function () { dragging = false; });
  });
  const wrap = root.closest(".layered-wrap") || root.parentNode.parentNode;
  function imgByZ(z) { return root.querySelector('.layer-img[data-z="' + z + '"]'); }
  // 显隐
  wrap.querySelectorAll(".layer-vis").forEach(function (cb) {
    cb.addEventListener("change", function () {
      const img = imgByZ(cb.dataset.z); if (img) img.style.display = cb.checked ? "" : "none";
    });
  });
  // 透明度
  wrap.querySelectorAll(".layer-opacity").forEach(function (sl) {
    sl.addEventListener("input", function () {
      const img = imgByZ(sl.dataset.z); if (img) img.style.opacity = sl.value / 100;
    });
  });
  // z 顺序 ±1
  function bump(z, d) {
    const img = imgByZ(z); if (!img) return;
    img.style.zIndex = Math.max(0, (parseInt(img.style.zIndex || "0", 10) + d));
  }
  wrap.querySelectorAll(".layer-up").forEach(function (b) {
    b.addEventListener("click", function () { bump(b.dataset.z, 1); });
  });
  wrap.querySelectorAll(".layer-down").forEach(function (b) {
    b.addEventListener("click", function () { bump(b.dataset.z, -1); });
  });
}
// 监听 Gradio 把新 HTML 塞进来
new MutationObserver(function () {
  document.querySelectorAll(".layered-canvas").forEach(initLayeredCanvas);
}).observe(document.body, { childList: true, subtree: true });
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".layered-canvas").forEach(initLayeredCanvas);
});
</script>
"""
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest web/tests/test_ui.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add web/frontend/ui.py web/tests/test_ui.py
git commit -m "feat(web): add stacked-canvas html and drag/layer-panel js"
```

---

### Task 9: `frontend/app.py` — Gradio Blocks 装配（手动验证）

**Files:**
- Create: `web/frontend/app.py`

无单元测试（Gradio 装配靠手动验收，见 Task 11）。

- [ ] **Step 1: 实现 app.py**

```python
# web/frontend/app.py
"""Gradio 前端：选模型 -> 传图 -> 调后端 -> 叠加画布 + 图层面板（可拖拽）+ 打包下载。
绑定 0.0.0.0:<frontend port>，靠 VSCode 端口转发访问。"""
from __future__ import annotations

import io
import os
import zipfile

import gradio as gr
from PIL import Image

from web.common.config import load_config
from web.common.imaging import decode_b64_png
from web.frontend import client
from web.frontend.ui import canvas_html, DRAG_JS

CFG = load_config(os.environ.get("CONFIG_PATH"))
MODELS = ["qwen-image-layered", "layerd"]


def _base_url(model: str) -> str:
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    return f"http://{host}:{CFG['ports'][model]}"


def run_infer(model, image, layers, steps, resolution, max_iterations):
    if image is None:
        raise gr.Error("请先上传一张图片")
    params = ({"layers": int(layers), "steps": int(steps), "resolution": int(resolution)}
              if model == "qwen-image-layered" else {"max_iterations": int(max_iterations)})
    try:
        resp = client.predict(_base_url(model), Image.fromarray(image), params, timeout=600)
    except Exception as e:
        raise gr.Error(f"后端推理失败：{e}")
    return canvas_html(resp), resp


def make_zip(resp):
    if not resp or not resp.get("layers"):
        raise gr.Error("还没有可下载的分层结果")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for l in resp["layers"]:
            png = decode_b64_png(l["png_b64"])
            b = io.BytesIO(); png.save(b, format="PNG")
            zf.writestr(f"L{l['z_order']}.png", b.getvalue())
    path = "/tmp/layers.zip"
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return path


def build() -> gr.Blocks:
    with gr.Blocks(head=DRAG_JS, title="分层模型部署") as demo:
        gr.Markdown("## 图像分层模型部署\n选择模型 → 上传图片 → 运行；分层结果叠加在画布上，可拖拽。")
        with gr.Row():
            model = gr.Dropdown(MODELS, value=MODELS[0], label="模型")
            image = gr.Image(label="输入图片", type="numpy")
        with gr.Accordion("高级参数", open=False):
            layers = gr.Slider(2, 20, value=CFG["models"]["qwen-image-layered"]["layers"], step=1, label="qwen 图层数")
            steps = gr.Slider(10, 60, value=CFG["models"]["qwen-image-layered"]["steps"], step=1, label="qwen 步数")
            resolution = gr.Slider(384, 1024, value=CFG["models"]["qwen-image-layered"]["resolution"], step=64, label="qwen 分辨率")
            max_iterations = gr.Slider(1, 10, value=CFG["models"]["layerd"]["max_iterations"], step=1, label="layerd 最大迭代")
        run = gr.Button("运行", variant="primary")
        out_html = gr.HTML()
        state = gr.State()
        dl = gr.Button("打包下载 zip")
        dl_file = gr.File(label="下载")

        run.click(run_infer, [model, image, layers, steps, resolution, max_iterations], [out_html, state])
        dl.click(make_zip, state, dl_file)
    return demo


def main():
    port = int(os.environ.get("PORT", CFG["ports"]["frontend"]))
    build().launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟 import（确认无语法/导入错误，需先装 gradio）**

Run: `.venv/bin/python -c "from web.frontend.app import build; print('app ok')"`
Expected: 打印 `app ok`（不启动服务）

- [ ] **Step 3: Commit**

```bash
git add web/frontend/app.py
git commit -m "feat(web): add gradio frontend app wiring"
```

---

### Task 10: 启停脚本 + README

**Files:**
- Create: `web/scripts/run_backend_qwen.sh`, `web/scripts/run_backend_layerd.sh`, `web/scripts/run_frontend.sh`, `web/scripts/stop_all.sh`
- Create: `web/README.md`

- [ ] **Step 1: 写 `run_backend_qwen.sh`**

```bash
#!/usr/bin/env bash
# 起 qwen 后端容器：host 选卡 + 量空闲显存 -> 传入容器；模型懒加载，空闲卸载。
set -euo pipefail
REPO_ROOT=/dev_share/zbchu2/autoexplore
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/runs/design-layer-may26}
TAG=${TAG:-autoexplore/qwen-image-layered}
PORT=${PORT:-8801}
NAME=web-backend-qwen

GPU=$(${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/scripts/gpu_select.py --count 1 --min-free-mib 18000)
FREE_MIB=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "${GPU}" \
           | awk -F', ' '{print $1-$2}')
echo "qwen backend: GPU=${GPU} FREE_MIB=${FREE_MIB} PORT=${PORT}"

docker rm -f "${NAME}" 2>/dev/null || true
INNER="unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/app && \
  python -m web.backend.server"

docker run -d --name "${NAME}" \
  --runtime=nvidia --gpus "device=${GPU}" \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -v "${RUN_DIR}:/work:ro" \
  -v "${REPO_ROOT}/web:/app/web:ro" \
  -v "${REPO_ROOT}/caches/modelscope:/root/.cache/modelscope:ro" \
  -e MODEL=qwen-image-layered -e PORT="${PORT}" -e FREE_MIB="${FREE_MIB}" \
  -e RUN_DIR_IN_CONTAINER=/work \
  -e MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen-Image-Layered \
  -e CONFIG_PATH=/app/web/config.yaml -e HOME=/tmp \
  "${TAG}" bash -c "${INNER}"
echo "started ${NAME}; logs: docker logs -f ${NAME}"
```

- [ ] **Step 2: 写 `run_backend_layerd.sh`**

```bash
#!/usr/bin/env bash
# 起 layerd 后端容器。layerd 轻量，单卡足够。
set -euo pipefail
REPO_ROOT=/dev_share/zbchu2/autoexplore
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/runs/design-layer-may26}
TAG=${TAG:-autoexplore/layerd}
PORT=${PORT:-8802}
NAME=web-backend-layerd

GPU=$(${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/scripts/gpu_select.py --count 1 --min-free-mib 10000)
FREE_MIB=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "${GPU}" \
           | awk -F', ' '{print $1-$2}')
echo "layerd backend: GPU=${GPU} FREE_MIB=${FREE_MIB} PORT=${PORT}"

docker rm -f "${NAME}" 2>/dev/null || true
INNER="unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=/tmp/hf_modules PYTHONPATH=/app && \
  mkdir -p /tmp/hf_modules && python -m web.backend.server"

docker run -d --name "${NAME}" \
  --runtime=nvidia --gpus "device=${GPU}" \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -v "${RUN_DIR}:/work:ro" \
  -v "${REPO_ROOT}/web:/app/web:ro" \
  -v "${HOME}/.cache/huggingface:/cache/hf:ro" \
  -v "${HOME}/.cache/torch:/cache/torch:ro" \
  -e MODEL=layerd -e PORT="${PORT}" -e FREE_MIB="${FREE_MIB}" \
  -e RUN_DIR_IN_CONTAINER=/work -e HF_CACHE=/cache/hf -e TORCH_HOME=/cache/torch \
  -e CONFIG_PATH=/app/web/config.yaml -e HOME=/tmp \
  "${TAG}" bash -c "${INNER}"
echo "started ${NAME}; logs: docker logs -f ${NAME}"
```

- [ ] **Step 3: 写 `run_frontend.sh`**

```bash
#!/usr/bin/env bash
# 起 Gradio 前端（host .venv），绑定 0.0.0.0:7860，靠 VSCode 端口转发访问。
set -euo pipefail
REPO_ROOT=/dev_share/zbchu2/autoexplore
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export CONFIG_PATH="${REPO_ROOT}/web/config.yaml"
exec .venv/bin/python -m web.frontend.app
```

- [ ] **Step 4: 写 `stop_all.sh`**

```bash
#!/usr/bin/env bash
# 停后端容器（前端是前台进程，Ctrl-C 即停）。
set -euo pipefail
docker rm -f web-backend-qwen web-backend-layerd 2>/dev/null || true
echo "stopped backend containers"
```

- [ ] **Step 5: chmod + 写 README**

```bash
chmod +x web/scripts/*.sh
```

`web/README.md`:
```markdown
# 分层模型 Web 部署

选模型 → 传图 → 得到叠加在一张画布上、可拖拽的 RGBA 分层结果。前端 Gradio（host）+ 每模型一个后端容器。

## 启动

```bash
# 1) 起后端（各自挑一张空闲卡，模型懒加载，空闲 600s 卸载）
web/scripts/run_backend_qwen.sh
web/scripts/run_backend_layerd.sh

# 2) 起前端（前台；Ctrl-C 退出）
web/scripts/run_frontend.sh
```

前端监听 `0.0.0.0:7860`。在 VSCode 里转发 7860 端口，本地浏览器打开 `http://localhost:7860`。

## 停止

```bash
web/scripts/stop_all.sh   # 停后端容器；前端 Ctrl-C
```

## 说明
- qwen 按所选卡空闲显存自动选 bf16(≥40GB)/int8 量化，推理后强制丢空层。
- 高级面板可调 qwen 的 layers/steps/分辨率、layerd 的 max_iterations。
- 分层图在画布上可拖拽，右侧面板可显隐/调 z 顺序/调透明度/单层或打包下载。
- 后端仅绑 127.0.0.1，外部不可达；只暴露前端 7860。

## 测试
```bash
.venv/bin/pytest web/tests            # 单元测试（CPU）
.venv/bin/pytest web/tests -m gpu     # GPU 冒烟（需 Docker+GPU+权重缓存）
```
```

- [ ] **Step 6: Commit**

```bash
git add web/scripts web/README.md
git commit -m "feat(web): add start/stop scripts and readme"
```

---

### Task 11: 后端 GPU 冒烟测试（默认 skip）+ 端到端手动验收

**Files:**
- Create: `web/tests/test_backend_smoke.py`

- [ ] **Step 1: 写 gpu-marked 冒烟测试**

```python
# web/tests/test_backend_smoke.py
"""真实后端冒烟：起 qwen 后端容器 -> /health -> /predict 一张小图 -> 断言返回层。
需 Docker + GPU + 权重缓存；默认 skip（pyproject: -m 'not gpu'）。"""
import base64
import io
import subprocess
import time

import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.gpu

PORT = 8801
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def backend():
    subprocess.run(["web/scripts/run_backend_qwen.sh"], check=True,
                   env={"PORT": str(PORT), "PATH": "/usr/bin:/bin"})
    # 等 http 起来（不等模型加载，模型在首个 /predict 才载）
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(2)
    yield BASE
    subprocess.run(["web/scripts/stop_all.sh"], check=False)


def test_health_then_predict_returns_layers(backend):
    h = requests.get(f"{backend}/health", timeout=5).json()
    assert h["model"] == "qwen-image-layered"
    buf = io.BytesIO()
    Image.new("RGB", (320, 320), (180, 200, 220)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(f"{backend}/predict",
                      json={"image_b64": b64, "params": {"layers": 4, "steps": 10}},
                      timeout=600)
    assert r.status_code == 200, r.text
    resp = r.json()
    assert len(resp["layers"]) >= 1
    assert resp["dropped"] >= 0
    assert resp["canvas"]["w"] == 320
```

- [ ] **Step 2: 确认默认跳过**

Run: `.venv/bin/pytest web/tests/test_backend_smoke.py -v`
Expected: `1 deselected`（因 `-m 'not gpu'`），不实际起容器。

- [ ] **Step 3: 跑全部单元测试**

Run: `.venv/bin/pytest web/tests -v`
Expected: 全绿（imaging 2 + postprocess 6 + config 4 + runners 4 + assemble 1 + manager 4 + client 4 + ui 6 = 31 passed，backend_smoke deselected）

- [ ] **Step 4: 手动端到端验收（人工执行，勾选记录结果）**

1. `web/scripts/run_backend_qwen.sh` 与 `run_backend_layerd.sh`，`docker logs` 确认 `[backend] ...` 打印。
2. `web/scripts/run_frontend.sh`，VSCode 转发 7860，浏览器打开。
3. 选 qwen、传一张真实设计图、运行：确认画布显示叠加结果、`dropped` 数 > 0、保留层数合理。
4. 拖拽某层 → 位置改变；勾显隐 → 该层显隐；调透明度滑块 → 该层变透明；↑/↓ → z 顺序变化。
5. 单层「下载」与「打包下载 zip」都能下载到 PNG。
6. 切到 layerd 重跑：返回较少层（约 3-4），交互同样可用。
7. 静置 > 10 分钟后再次运行：确认能重新加载模型（idle 卸载后重载）。

- [ ] **Step 5: Commit**

```bash
git add web/tests/test_backend_smoke.py
git commit -m "test(web): add gpu-marked backend smoke test"
```

---

## 自查（Self-Review）

- **Spec 覆盖**：模型选择(app Dropdown)✓、传图(app Image)✓、叠加画布(ui canvas_html)✓、拖拽+显隐+zorder+透明度(DRAG_JS)✓、单层/打包下载(ui + app make_zip)✓、qwen 丢空层(postprocess + assemble，强制)✓、按需加载+空闲卸载(ModelManager)✓、qwen bf16/int8 自动(choose_qwen_dtype + 脚本传 FREE_MIB)✓、两镜像分离的前后端(server/client/scripts)✓、stdlib http.server✓、端口 127.0.0.1 后端 + 0.0.0.0 前端✓、高级面板可调参(app Accordion)✓、复用既有 infer.py(runners 反射 import)✓、web/ 隔离✓、gpu-mark 测试约定✓。
- **占位符**：无 TBD/TODO；每个代码步骤含完整代码。
- **类型/命名一致**：`encode_png_b64`/`decode_b64_png`/`coverage`/`alpha_bbox`/`drop_empty`/`choose_qwen_dtype`/`build_runner`/`assemble_response`/`ModelManager.get|maybe_unload|loaded`/`client.predict|health`/`canvas_html`/`DRAG_JS` 在定义与调用处一致。
- **可选增强（不在本计划）**：拖拽排布后的「拼合导出」、layerd 的 drop-empty 阈值 UI——spec 已列为后续。
