#!/usr/bin/env bash
# 停所有后端容器（前端是前台进程，Ctrl-C 即停）。
# 按部署的模型把 web-backend-<model> 列全，或用通配前缀批量删。
set -euo pipefail

# 方式一：显式列名（把 model-a/model-b 替换成你的模型名）
docker rm -f "web-backend-model-a" "web-backend-model-b" 2>/dev/null || true

# 方式二：按前缀批量（一次清掉本部署所有后端）
# docker ps -aq --filter "name=^web-backend-" | xargs -r docker rm -f

echo "stopped backend containers"
