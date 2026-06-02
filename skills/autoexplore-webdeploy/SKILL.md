---
name: autoexplore-webdeploy
description: Use when deploying a reproduced/optimized model (a phase1/phase2 `infer.py` under some `runs/<tag>/`) as an interactive web app — drives a model-agnostic Gradio frontend + per-model backend container, reusing infer.py rather than rewriting inference.
---

# autoexplore Web 部署:把已复现模型部署成可交互 web

输入:某个 `runs/<tag>/` 下已有的 `infer.py`(phase1 复现 / phase2 优化的产物)。
产出:host 上一个 Gradio 前端 + 每模型一个后端 Docker 容器,支持选模型 → 传输入 → 跑推理 → 渲染输出 → 可选下载。

环境:云 GPU 服务器,Docker + NVIDIA runtime,按空闲度选卡;前端靠 VSCode 端口转发访问(单用户内网)。

**这是通用 skill,不绑定任何具体模型、领域或输出模态。** 它只给「模型 web 部署」的可复用模式与纪律;一切模型特例(包哪个 infer.py、参数集、输出怎么渲染)都是部署时按模板填空的内容。

## 架构

- **前端**(host `.venv`):Gradio,绑 `0.0.0.0:<port>`。选模型 → 传输入 → 调后端 → 渲染输出。
- **后端**(每模型一个容器):stdlib `http.server`,绑 `127.0.0.1:<port>`(外部不可达)。模型**懒加载**进 GPU、空闲 `idle_ttl` 秒后卸载。重模型镜像里**不塞 web 框架**(复用 phase1 复现镜像)。
- **复用而非重写**:后端 Runner 适配器用 `_import_from` 薄封装 infer.py 的加载/推理入口——这是与 phase1/phase2 的接缝,**不重写推理**。
- **契约**:请求 `{<input-key>_b64, params}`;响应 `{model, infer_ms, <模态输出>, ...}`。

## 核心抽象:通用基建 vs 部署特例

skill 的灵魂是切开二者。**通用基建**(模板已出,几乎不改):stdlib server 循环 + `/health` `/predict`、懒加载管理器 + idle TTL 卸载 + 线程锁、Runner 适配协议、HTTP client、config.yaml 模式、启停脚本。**部署特例**(落地时填的 5 个 seam):包哪个 infer.py、`build_runner`/`Runner`、输入解码(SEAM 1)、输出装配(SEAM 2)、UI 参数控件(SEAM A)、输出渲染器(SEAM B)。

详见 [references/deployment-pattern.md](references/deployment-pattern.md):目录落地形态、五个 seam、请求/响应契约、通用 common 层、输出渲染器指南、纪律全表、端到端验证清单。

## 模板

`references/templates/` 是骨架,**不要求 import 即跑**;每处「按模型填」均有标注。落地时拷进部署仓库的 `web/` 包再填 seam。

| 模板 | 用途 |
|------|------|
| `backend_server.py` | 通用 stdlib server + `ModelManager`(懒加载/idle 卸载);两个 SEAM 留空 |
| `runner_adapter.py` | `Runner` 协议 + `_import_from` + `build_runner` 工厂 + 1 个填空示例 |
| `frontend_app.py` | 通用 Gradio 骨架;`collect_params`/`render_output` 留空 |
| `config.yaml` | 配置模式样例(ports / idle_ttl / 每模型参数) |
| `run_backend.sh` | 后端容器模板(选卡、缓存从 `REPO_ROOT/caches` 挂载、代理处理) |
| `run_frontend.sh` | 前端模板(`0.0.0.0` + `PORT=` 覆盖 + 代理处理) |
| `stop_all.sh` | 停后端容器 |

## 关键纪律(详表见 references)

- **Cache 落项目目录**:权重下载到 `<repo>/caches/{modelscope,huggingface,torch}`,`:ro` 挂载,env 注入 `HF_HOME`/`MODELSCOPE_CACHE`/`TORCH_HOME`;**绝不依赖 `~/.cache`**(`HOME=/tmp`)。
- **容器**:`--runtime=nvidia`(否则 torch.cuda 失效)、`--user $UID:$GID`(否则产物 root 所有)、按空闲显存选卡(复用 phase1 `gpu_select.py`)。
- **代理**:容器内 unset 全套 `*_proxy`;host curl 的 `NO_PROXY` 必须含 `127.0.0.1`(不只 `localhost`),否则被 SOCKS/tinyproxy 劫持 500。
- **共享机端口**:默认口可能被别的租户占,先 `ss -ltn` 探活再选空闲口,前端 `PORT=` 覆盖。
- **UI 取值域**:控件只暴露模型真实接受的值(离散值用 `Radio`/`Dropdown`,别用 `Slider` 放进非法值);`merge_params` 按 `_INFER_KEYS` 裁剪是第二道防线。
- **idle_ttl**:必须大于最坏单次推理耗时,否则长推理被中途卸载。

## 第一版明确不做(避免 scope 膨胀)

- 多机/多副本/负载均衡——单机、每模型单容器。
- 鉴权 / 多用户隔离——单用户内网(前端靠 VSCode 端口转发)。
- 模板独立可跑——是骨架,标注「此处按模型填」。
- 把任何具体领域语义(如图像分层画布)写进模板——只在文档里作为「一种输出渲染器」举例。

## 验证门(两端都过才算完成)

部署后跑 [references/deployment-pattern.md](references/deployment-pattern.md) 的端到端清单:host 单测 → 后端 `/health` → `/predict` 小输入断言契约 → 前端转发跑通渲染 → idle 卸载复核。任一端不过都不算完成。
