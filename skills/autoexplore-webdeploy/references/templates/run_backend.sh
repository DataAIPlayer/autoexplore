#!/usr/bin/env bash
# 通用后端容器启动模板：host 选卡 + 量空闲显存 -> 注入容器；模型懒加载、空闲卸载。
# **按模型填空**：下方 <...> 占位（MODEL、TAG、缓存挂载、权重路径）替换成你的真实值。
# 复制成 run_backend_<model>.sh（一模型一份），或用 MODEL= 环境变量参数化。
set -euo pipefail

# ── 路径基建（通用）────────────────────────────────────────────────────────────
REPO_ROOT=${REPO_ROOT:-/path/to/repo}                 # 提供 .venv / runs / caches / scripts
WEB_DIR=$(cd "$(dirname "$0")/.." && pwd)             # web/ 目录（worktree 或主仓库均可）
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/runs/TAG_FILL}        # phase1/phase2 产出的 run 目录，含 infer.py
MODEL=${MODEL:-model-a}                                # 模型名（与 config.yaml / build_runner 一致）
TAG=${TAG:-autoexplore/model-a}                       # 该模型的 docker 镜像 tag（phase1 复现时构建）
PORT=${PORT:-8801}
NAME=web-backend-${MODEL}

# ── 选卡 + 量空闲显存（通用，复用 phase1 的 gpu_select.py）──────────────────────
GPU=$("${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/gpu_select.py" --count 1 --min-free-mib 10000)
FREE_MIB=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "${GPU}" \
           | awk -F', ' '{print $1-$2}')
echo "${MODEL} backend: GPU=${GPU} FREE_MIB=${FREE_MIB} PORT=${PORT}"

docker rm -f "${NAME}" 2>/dev/null || true

# ── 容器内启动命令 ──────────────────────────────────────────────────────────────
# 纪律：unset 全套 *_proxy（容器内本地推理不该走代理）；离线加载权重（HF_HUB_OFFLINE 等）。
# trust_remote_code 模型需要 HF_MODULES_CACHE 指向容器内可写位置。
INNER="unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=/tmp/hf_modules PYTHONPATH=/app && \
  mkdir -p /tmp/hf_modules && python -m web.backend.server"

# ── docker run（通用纪律 + 按模型填的缓存挂载/权重路径）──────────────────────────
# 纪律：--runtime=nvidia（否则 torch.cuda 失效）；--user $UID:$GID（否则产物 root 所有）；
#       端口绑 127.0.0.1（外部不可达）；RUN_DIR 与 WEB_DIR 都 :ro 挂载。
# **Cache 落项目目录**：权重缓存从 ${REPO_ROOT}/caches/* 挂载，env 注入 HF_HOME/MODELSCOPE_CACHE/
# TORCH_HOME 指向容器内挂载点；**绝不依赖 ~/.cache**（HOME=/tmp 强制隔离）。
docker run -d --name "${NAME}" \
  --runtime=nvidia --gpus "device=${GPU}" \
  --user "$(id -u):$(id -g)" \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -v "${RUN_DIR}:/work:ro" \
  -v "${WEB_DIR}:/app/web:ro" \
  -v "${REPO_ROOT}/caches/huggingface:/cache/huggingface:ro" \
  -v "${REPO_ROOT}/caches/modelscope:/cache/modelscope:ro" \
  -v "${REPO_ROOT}/caches/torch:/cache/torch:ro" \
  -e MODEL="${MODEL}" -e PORT="${PORT}" -e FREE_MIB="${FREE_MIB}" \
  -e RUN_DIR_IN_CONTAINER=/work \
  -e HF_HOME=/cache/huggingface -e MODELSCOPE_CACHE=/cache/modelscope -e TORCH_HOME=/cache/torch \
  -e MODEL_PATH="CONTAINER_WEIGHTS_PATH_FILL_PER_MODEL" \
  -e CONFIG_PATH=/app/web/config.yaml -e HOME=/tmp \
  "${TAG}" bash -c "${INNER}"
echo "started ${NAME}; logs: docker logs -f ${NAME}"
