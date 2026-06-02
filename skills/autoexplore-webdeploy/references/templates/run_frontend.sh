#!/usr/bin/env bash
# 通用前端启动模板：起 Gradio（host .venv），绑 0.0.0.0:<port>，靠 VSCode 端口转发访问。
# 通常无需按模型改——模型差异都在 config.yaml / frontend_app.py 的 seam 里。
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/repo}                        # 提供 .venv
CHECKOUT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)           # web/ 的父目录（含 web 包的根）
cd "${CHECKOUT_ROOT}"
export PYTHONPATH="${CHECKOUT_ROOT}"
export CONFIG_PATH="${CHECKOUT_ROOT}/web/config.yaml"

# 纪律（代理）：gradio/httpx 在导入时会读 SOCKS 代理而报错；调本机后端走 localhost 也不该过代理。
# NO_PROXY 必须同时含 localhost 和 **127.0.0.1**（后端绑 127.0.0.1；只写 localhost 会被
# SOCKS/tinyproxy 劫持成 500）。PORT= 可覆盖端口（共享机默认口可能被占，先 ss -ltn 探活）。
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

exec "${REPO_ROOT}/.venv/bin/python" -m web.frontend.app
