"""通用模型适配层模板:把 phase1/phase2 复现产出的 `infer.py` 的加载/推理逻辑，
薄封装成统一 Runner 协议，供 backend_server 懒加载调用。

**核心原则:复用而非重写。** 不在这里实现推理；只 `_import_from` 已有 infer.py 的入口
（load_*、infer_one/run/decompose 等），适配成 load()/infer()/unload() 三个方法。
torch 等重依赖只在 load() 内 import（只有容器里才装了）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

# 容器内 RUN_DIR 挂载点（run_backend.sh 用 -v RUN_DIR:/work）；infer.py 就在它下面。
RUN_DIR = Path(os.environ.get("RUN_DIR_IN_CONTAINER", "/work"))


def _import_from(dir_path: Path, module_name: str):
    """把 infer.py 所在目录加入 sys.path 后 import 它。
    NOTE: module_name 进 sys.modules 缓存；一个容器=一个模型，故安全。若某天两个
    Runner 要同进程共存，改用 importlib.util.spec_from_file_location 按文件加载。"""
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))
    return __import__(module_name)


@runtime_checkable
class Runner(Protocol):
    """每个模型实现这四样。infer 的输入/输出类型随模态而定（与 server 的两个 SEAM 对齐）。"""
    def load(self) -> None: ...
    def infer(self, model_input, params: dict): ...
    def unload(self) -> None: ...
    def health_meta(self) -> dict: ...   # /health 展示的轻量元信息，如 {"dtype": "bf16"}


# ─────────────────────────────────────────────────────────────────────────────
# 示例 Runner（填空骨架，非可跑代码）。复制改名，把 <...> 换成你模型的真实入口。
# ─────────────────────────────────────────────────────────────────────────────
class ExampleRunner:
    name = "<your-model>"

    def __init__(self, cfg: dict, free_mib: int = 0):
        self._dir = RUN_DIR / "models" / self.name      # infer.py 所在目录
        self._handle = None                              # 加载后的 pipe/model 句柄
        # 按需从 free_mib/cfg 决定精度、量化等部署变体（可选）。
        self._dtype = "bf16"

    @property
    def is_loaded(self) -> bool:
        return self._handle is not None

    def load(self) -> None:
        infer = _import_from(self._dir, "infer")         # 复用已有 infer.py
        self._handle = infer.load_pipeline(os.environ["MODEL_PATH"])  # <- 换成真实加载入口
        self._run = infer.infer_one                      # <- 换成真实推理入口

    def infer(self, model_input, params: dict):
        # 把 params 里的键按 infer 入口签名展开；不要在这里重写推理逻辑。
        return self._run(self._handle, model_input, **params)

    def unload(self) -> None:
        import torch
        self._handle = None
        torch.cuda.empty_cache()

    def health_meta(self) -> dict:
        return {"dtype": self._dtype}


def build_runner(model: str, free_mib: int, cfg: dict) -> Runner:
    """模型名 -> Runner 实例的工厂。每多一个模型加一条分支。"""
    if model == ExampleRunner.name:
        return ExampleRunner(cfg=cfg, free_mib=free_mib)
    raise ValueError(f"unknown model: {model}")
