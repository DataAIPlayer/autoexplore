"""通用单模型后端模板:stdlib http.server 暴露 /health 与 /predict；模型懒加载进 GPU、
空闲 TTL 卸载。**与模型/模态无关**——三个 seam 由具体部署填空（见下方 SEAM 标注）。

启动通过环境变量配置：MODEL、PORT、IDLE_TTL、FREE_MIB（以及部署自定义的 env）。
不要在这里 import torch/diffusers/任何重依赖；它们只在 Runner.load() 里、容器内 import。
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 部署提供：build_runner(model, **kw) -> Runner（见 runner_adapter.py）
from runner_adapter import build_runner
# 部署提供：config 读取 + 按模型裁剪参数（见 config.yaml 模式；通常放 common/config.py）
from config import load_config, merge_params


# ─────────────────────────────────────────────────────────────────────────────
# SEAM 1 —— 输入解码：把请求里的 input_b64 解成模型 infer() 接受的对象。
#   图像模态：base64 PNG -> PIL.Image。文本/音频/其它模态：按需替换。
# SEAM 2 —— 输出装配：把 Runner.infer() 的原始输出装成 JSON 可序列化的响应契约。
#   两者通常抽到 common/ 模块；这里以函数占位，标明这是“按模态填”的部分。
# ─────────────────────────────────────────────────────────────────────────────
def decode_request(req: dict):
    """返回 (model_input, params)。params 已按模型裁剪。"""
    raise NotImplementedError("SEAM 1: 按模态解码 req['input_b64'] -> model_input")


def assemble_response(model: str, meta: dict, raw_output) -> dict:
    """返回 JSON 可序列化的 dict，形如 {model, infer_ms, <模态输出>, ...}。"""
    raise NotImplementedError("SEAM 2: 按模态把 raw_output 装配成响应")


class ModelManager:
    """懒加载 + 空闲 TTL 卸载 + 线程锁。**通用，无需改。**"""

    def __init__(self, runner_factory, idle_ttl: float = 600.0):
        self._factory = runner_factory
        self._runner = None
        self._last_used = None
        self._lock = threading.Lock()
        self.idle_ttl = idle_ttl

    @property
    def loaded(self) -> bool:
        with self._lock:
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


def _make_handler(model: str, manager: ModelManager, cfg: dict, meta_hint: dict):
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
                self._send(200, {"loaded": manager.loaded, "model": model, **meta_hint})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                req["params"] = merge_params(model, req.get("params", {}), cfg)  # 只留模型认的键
                model_input, params = decode_request(req)            # SEAM 1
                runner = manager.get(now=time.time())
                t0 = time.time()
                raw = runner.infer(model_input, params)
                infer_ms = int((time.time() - t0) * 1000)
                meta = {"infer_ms": infer_ms, **meta_hint}
                self._send(200, assemble_response(model, meta, raw))  # SEAM 2
            except Exception as e:  # 兜底：返回 JSON 错误而非崩连接
                hint = "GPU 显存不足，换更小参数。" if "out of memory" in str(e).lower() else ""
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

    def factory():
        return build_runner(model, free_mib=free_mib, cfg=cfg)

    # /health 展示用的元信息（如 dtype）；factory() 只构造、不 load()，无 GPU 开销。
    meta_hint = {"free_mib": free_mib, **factory().health_meta()}
    manager = ModelManager(factory, idle_ttl=idle_ttl)

    stop = threading.Event()
    threading.Thread(target=_idle_loop, args=(manager, stop), daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(model, manager, cfg, meta_hint))
    httpd.daemon_threads = True  # 别让在途 GPU 线程卡住 Ctrl-C
    print(f"[backend] model={model} port={port} idle_ttl={idle_ttl} meta={meta_hint}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
