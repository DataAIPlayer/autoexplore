"""冻结测速协议:动态加载 adapter,对 bench_config 钉死的固定子集做 batch=1 单条计时,
一次执行同产 speed.json(单条延迟均值/p50/p99 + 吞吐)与 predictions.jsonl(phase1 schema)。
本脚本不含任何框架/任务语义——那些都在 Claude 写的 adapter.py 里。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


def load_adapter(path: Path):
    """从路径动态加载 adapter 模块;须暴露 load_model() 与 infer_one()。"""
    spec = importlib.util.spec_from_file_location("p3_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_model") or not hasattr(mod, "infer_one"):
        raise ValueError("adapter 必须定义 load_model() 与 infer_one()")
    return mod


def _record_id(s: dict) -> str:
    return s.get("image_id", s.get("id"))


def select_records(bench_config: dict, dataset_dir: Path) -> list[dict]:
    """固定记录子集:有 subset_ids 按其顺序取;否则按 id 排序取前 n_records(默认全量)。"""
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    samples = meta["samples"]
    ids = bench_config.get("subset_ids")
    if ids:
        by = {_record_id(s): s for s in samples}
        return [by[i] for i in ids]
    n = int(bench_config.get("n_records", len(samples)))
    return sorted(samples, key=_record_id)[:n]
