# 分层模型 Web 部署 — 设计文档

日期：2026-05-29
分支：optimize/design-layer-may26
目标产物：`web/`（项目根目录下，独立于 autoexplore 智能体/实验循环的产物）

## 1. 目标与范围

为 `runs/design-layer-may26` 里两个已就绪的图像分层模型做一个 Gradio web 部署：

- 用户在 UI 上**选择模型**（`qwen-image-layered` / `layerd`）。
- 上传一张图片 → 后端推理 → 返回一组 RGBA 分层图。
- 分层图**叠加在同一张画布上**显示，且**每一层可自由拖拽改变位置**；右侧图层面板支持显隐、调 z 顺序、调透明度、单层/打包下载。
- qwen 推理后**必须做丢空层后处理**（qwen 固定输出多层，常含空层）。

**非目标 / YAGNI**：不做用户账号、不做历史记录持久化、不做并发多用户排队（单用户/小规模演示即可）、不做模型训练或指标评估（那是 autoexplore 智能体的事）。

## 2. 约束（来自用户与环境）

1. 所有代码隔离在项目根的 `web/`，不碰 `runs/`、`skills/` 等智能体产物。
2. 两个模型分属**两个不同 Docker 镜像**（`autoexplore/qwen-image-layered`、`autoexplore/layerd`，依赖不同），单进程无法同时 import → 必须前后端分离、每模型一个后端。
3. 运行在远程云服务器，用户用 VSCode 远程控制 → Gradio 绑定 `0.0.0.0:<固定端口>`，靠 VSCode 端口转发访问；**不用** `share=True` 公网隧道（代理环境不稳）。
4. 共享 GPU 集群、多 GPU：模型**按需加载 + 空闲超时卸载**，不长期霸占显存。
5. 两个镜像都基于 `qwen_image_edit:v2`（conda Python 3.11，torch 2.6）。

## 3. 架构

```
浏览器 (VSCode 转发 :7860)
        │  HTTP
   Gradio 前端  (host 的 .venv, web/frontend/app.py)
        │  HTTP POST /predict  (127.0.0.1:<backend port>)
   ┌────┴───────────────────────────────┐
   │                                    │
 qwen 后端容器                        layerd 后端容器
 镜像 autoexplore/qwen-image-layered   镜像 autoexplore/layerd
 web/backend/server.py (stdlib http.server)
 - 容器常驻、轻量；模型懒加载进 GPU，空闲 TTL 后卸载
 - 复用 runs/.../models/<m>/infer.py 的加载/推理函数
 挂载: /work(RUN_DIR :ro, 取 infer.py)  /web(web/ :ro)  权重缓存(:ro)
```

**为什么这样分层**：两镜像依赖互斥，前后端分离是硬约束下的唯一干净解；前端只依赖 `gradio/requests/pillow/numpy`，放 host .venv 最轻；后端用 stdlib `http.server` 避免给镜像加 web 框架依赖（无需重建镜像）。

### 3.1 各单元职责（一个单元一件事）

| 单元 | 文件 | 职责 | 依赖 |
|---|---|---|---|
| 前端应用 | `web/frontend/app.py` | 组装 Gradio Blocks、事件绑定、调后端、出错提示 | gradio, `client.py`, `ui.py`, `common/` |
| 后端客户端 | `web/frontend/client.py` | `predict(model, image, params) -> layers` 的 HTTP 封装、超时/重试 | requests |
| UI 构建 | `web/frontend/ui.py` | 由层列表生成「叠加画布 + 图层面板」HTML（data-URI 内嵌）；提供拖拽/显隐/zorder/透明度的前端 JS（经 `Blocks(head=...)` 注入 + MutationObserver 挂载） | 无（纯字符串/HTML） |
| 后端服务 | `web/backend/server.py` | `http.server`：`/health`、`/predict`；懒加载 + 空闲 TTL 卸载 + 加载锁 | stdlib, `runners.py`, `common/` |
| 模型适配 | `web/backend/runners.py` | 把已有 infer 逻辑包成统一接口 `Runner.load()/infer(img, params)/unload()`；qwen 按 free-mem 选 bf16/int8 | 反射式 import 现有 `infer.py` / `infer_q8.py` |
| 公共后处理 | `web/common/postprocess.py` | `coverage(rgba)` 与 `drop_empty(layers, thresh)`（复用 phase2 语义） | numpy, pillow |
| 启停脚本 | `web/scripts/*.sh` | 用 `gpu_select.py` 选卡、`docker run` 起后端、起前端、停全部 | bash, docker |
| 配置 | `web/config.yaml` | 端口、idle_ttl、各模型默认参数、丢空层阈值 | — |

## 4. 数据流与接口

### 4.1 后端 `/predict`

请求（JSON）：
```json
{ "image_b64": "<PNG base64>",
  "params": { "layers": 8, "steps": 30, "resolution": 640, "max_iterations": 6 } }
```
（前端按所选模型只发该模型用得到的字段；后端忽略无关字段并用默认值兜底。）

响应（JSON）：
```json
{ "model": "qwen-image-layered",
  "infer_ms": 12345,
  "dtype": "int8",                 // qwen 实际走的路径（auto 决策结果）
  "canvas": { "w": 640, "h": 853 },
  "layers": [
    { "z_order": 0, "png_b64": "...", "coverage": 0.42, "bbox": [x,y,w,h] }
  ],
  "dropped": 3 }                   // 后处理丢掉的空层数（qwen 关键信息）
```

错误：HTTP 4xx/5xx + `{ "error": "...", "hint": "..." }`（如 OOM 时 hint 建议降 layers / 用 int8）。

### 4.2 推理 → 后处理 → 装配

1. 后端按模型调用对应 `Runner.infer()` 得到全画布尺寸的 RGBA 层列表（back→front）。
2. **丢空层**：`drop_empty(layers, thresh)`（qwen 默认 thresh=0.0 仅丢真空层；至少保留覆盖度最高一层）。qwen 必做；layerd 也跑（已自停，基本无副作用）。
3. 每层 PNG 编码为 base64，连同 `coverage`、紧致 alpha `bbox`（用于拖拽初始把手，可选）一并返回。
4. 前端把各层作为 data-URI `<img>`，在 `.canvas` 容器里**绝对定位、(0,0) 叠放**，按 z_order 设 `z-index`。

### 4.3 画布交互（全部前端 JS，推理后无需再请求后端）

- **拖拽**：每个 `.layer-img` 监听 pointerdown/move/up，更新 `transform: translate(dx,dy)`。层是全画布尺寸（内容外透明），拖整张图等效于移动该层内容。
- **图层面板**（右侧，上=顶层）：每层一行 = 显隐 checkbox（toggle `display`）+ `↑/↓`（改 z-index）+ 透明度 slider（改 `opacity`）+ 单层下载（data-URI `<a download>`）。
- **打包下载**：Gradio 按钮 → 后端无关、前端 Python 把已持有的层打 zip → `gr.File` 下发。MVP 导出**原始层**；导出「拖拽排布后的拼合图」列为可选增强。
- JS 注入方式：`gr.Blocks(head=<script>)` 注册 window 上的初始化函数 + `MutationObserver` 监听 `gr.HTML` 内新出现的 `.canvas`，自动挂载拖拽/面板事件（规避 Gradio 对组件内联 `<script>` 的清洗）。

## 5. 后端懒加载 / 卸载 / GPU

- 容器随 web 服务常驻（不占 GPU，直到首个 `/predict`）。
- 首个请求触发 `Runner.load()`：加载到 `cuda`；记录 `last_used`。
- 后台守护线程每 30s 检查，`now - last_used > idle_ttl`（默认 600s）则 `unload()`：`del pipe; torch.cuda.empty_cache()`，回到未加载态。
- 加载/卸载用 `threading.Lock` 串行化，防止并发请求与卸载竞争。
- **GPU 选择**：启动脚本在 host 用 `scripts/gpu_select.py` 选 1 张卡，把 `device id` 与该卡 `free_mib` 通过 `--gpus device=<id>` 和环境变量传入容器。
- **qwen dtype 自动**：`runners.py` 读 `free_mib`，`>= 40000` 走基线 `infer.py.load_pipeline`（bf16），否则走 `inference_tuning/exp_layers16/infer_q8.py.load_pipeline`（torchao int8wo）。`infer_one` 复用基线实现。实际走的路径回填到响应 `dtype`。

## 6. 复用既有代码（不重复实现模型逻辑）

`runners.py` 把以下目录加入 `sys.path` 并 import 现成函数：
- qwen：`runs/design-layer-may26/models/qwen-image-layered/infer.py` 的 `load_pipeline / infer_one / fit_to_resolution`；int8 路径用 `phase2/inference_tuning/exp_layers16/infer_q8.py` 的 `load_pipeline`。
- layerd：`runs/design-layer-may26/models/layerd/infer.py` 的 `load_layerd`，调 `inst.decompose(img, max_iterations=...)`。

容器内通过挂载 `/work`（=RUN_DIR）可见这些文件。后处理 `coverage` 语义与 `phase2/postprocess_drop_empty.py` 一致（`common/postprocess.py` 抽出共享，前端 host 也直接 import 同一文件）。

后端需像现有脚本一样 `unset ALL_PROXY/...`、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，权重缓存（modelscope / huggingface / torch）以 `:ro` 挂载。

## 7. 端口与访问

| 服务 | 绑定 | 说明 |
|---|---|---|
| Gradio 前端 | `0.0.0.0:7860` | VSCode 转发此端口即可在本地浏览器打开 |
| qwen 后端 | `127.0.0.1:8801` | docker `-p 127.0.0.1:8801:8801`，仅本机可达 |
| layerd 后端 | `127.0.0.1:8802` | 同上 |

均在 `config.yaml` 可改。前端只需用户转发 7860 一个端口。

## 8. 错误处理

- 后端模型加载/推理异常 → 捕获，返回 JSON error；OOM 额外给 hint。
- 前端 `client.py` 超时（qwen 首次含加载，超时设大，如 600s）/连接失败 → `gr.Error` 弹窗，不崩 UI。
- `/health` 返回 `{loaded: bool, model, dtype, free_mib}`，前端启动时探活，离线后端在下拉里标灰并提示「未启动」。
- 上传非图片 / 过大 → 前端校验并提示。

## 9. 测试策略

- **单元（CPU，默认跑）**：`common/postprocess.py` 用合成 RGBA 数组验 `coverage`/`drop_empty`（含「全空→保留最高覆盖一层」边界）；`ui.py` 验 HTML/data-URI 装配（不起服务）；`client.py` 用 mock HTTP 验请求/错误路径。
- **后端冒烟（GPU，默认 skip，复用 phase2 既有 gpu-mark 约定）**：起一个后端，对一张小合成图打 `/predict`，断言返回层数>0、qwen `dropped>=0`、shape 一致。
- **手动验收**：真实图片走完整 UI，确认叠加显示、拖拽、显隐、zorder、透明度、单层与打包下载、模型切换、idle 卸载后再次请求能重载。

## 10. 文件布局

```
web/
  README.md              # 起停步骤、端口转发说明
  config.yaml            # 端口 / idle_ttl / 各模型默认参数 / drop-empty 阈值
  frontend/
    app.py               # Gradio Blocks + 事件 + head 注入 JS
    ui.py                # 画布/面板 HTML 构建 + 拖拽 JS 字符串
    client.py            # 后端 HTTP 客户端
  backend/
    server.py            # http.server: /health /predict + 懒加载/TTL/锁
    runners.py           # qwen / layerd 统一 Runner，复用现有 infer.py
  common/
    postprocess.py       # coverage + drop_empty（与 phase2 同语义）
  scripts/
    run_backend_qwen.sh  # gpu_select → docker run 起 qwen 后端
    run_backend_layerd.sh
    run_frontend.sh      # host .venv 起 gradio
    stop_all.sh
  tests/
    test_postprocess.py
    test_ui.py
    test_client.py
    test_backend_smoke.py  # gpu-marked, 默认 skip
```

## 11. 待实现时再定（不影响本设计）

- 拖拽排布后「拼合导出」是否要做（可选增强）。
- 是否给 layerd 也开 drop-empty 阈值 UI（默认开但 thresh=0）。
- 前端 host .venv 是否已装 gradio（若无，`uv pip install gradio` 走清华镜像，见 network 记忆）。
