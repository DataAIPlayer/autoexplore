# autoexplore 第一版设计 · Web 部署 skill(`autoexplore-webdeploy`)

**日期**: 2026-05-30
**范围**: 把"将一个已复现/已优化的模型部署成可交互 web"提炼为**通用、与模型/领域无关**的 Claude Code Skill
**形态**: 第三个 skill(继 `autoexplore-phase1` 复现、`autoexplore-phase2` 优化之后)——SKILL.md 纲领 + `references/` 模式文档 + 与模型无关的代码模板(混合)
**前置**: 某个 `runs/<tag>/` 下已有可用的 `infer.py`(phase1/phase2 产物);云 GPU + Docker + NVIDIA runtime

## 1. 目标与范围

构建一个**通用 web 部署 agent 能力**:输入一个已复现模型的 `infer.py`,产出 host 上的 Gradio 前端 + 每模型一个后端容器,支持选模型、传输入、跑推理、渲染输出。

**这是一个通用 skill,不绑定任何具体模型、领域或输出模态。** 它只描述"模型 web 部署"的可复用模式与纪律;一切模型特例(包哪个 `infer.py`、参数集、输出怎么渲染)都是部署时按模板填空的内容。当前本地 `web/`(qwen/layerd 图像分层实例,已转为未跟踪本地目录)只是**一个worked example**,其分层画布/拖拽/丢空层等均为图像分层领域特有,**不进入 skill**。

### 核心抽象:通用基建 vs 部署特例

skill 的灵魂是切开二者:

| 通用基建(skill 出模板) | 部署特例(部署时填) |
|---|---|
| stdlib `http.server` 后端循环 + `/health` `/predict` | 包哪个 `infer.py`、模型参数键 |
| 懒加载管理器 + idle TTL 卸载 + 线程锁 | 输入解码 / 输出装配(随模态) |
| Runner 适配器协议(`_import_from` 复用 infer.py,不重写推理) | UI 参数控件与取值域 |
| HTTP client、config.yaml 模式、请求/响应 JSON 契约 | 输出渲染器(图片 / 文本 / 交互画布 …) |
| 启动/停止脚本(选卡、缓存挂载、代理、端口) | |

### 第一版明确不做(避免 scope 膨胀)
- 多机/多副本/负载均衡——单机、每模型单容器。
- 鉴权 / 多用户隔离——单用户内网场景(前端靠 VSCode 端口转发)。
- 通用模板独立可跑——模板是骨架,标注"此处按模型填",不要求 import 即跑。
- 把任何图像分层语义写进模板——分层画布只在文档里作为"一种输出渲染器"举例。

## 2. 架构

- **前端**(host,`.venv`):Gradio,绑 `0.0.0.0:<port>`,靠 VSCode 端口转发访问。选模型 → 传输入 → 调后端 → 渲染输出 → 可选下载。
- **后端**(每模型一个 Docker 容器):stdlib `http.server`,绑 `127.0.0.1:<port>`(外部不可达)。模型**懒加载**进 GPU、空闲 `idle_ttl` 秒后卸载。重模型镜像里**不塞 web 框架**。
- **复用而非重写**:后端的 Runner 适配器用 `_import_from` 薄封装 phase1/phase2 产出的 `infer.py` 的 `load_*`/推理入口——这是与前两个 skill 的接缝。
- **契约**:请求 `{input_b64, params}`;响应 `{model, infer_ms, <模态相关输出>, ...}`。输出形状随模态而定,由部署的"输出装配"与"输出渲染器"两个 seam 决定。

## 3. 目录结构

```
skills/autoexplore-webdeploy/
  SKILL.md                          # 纲领:何时用、架构、基建/特例切分、契约、纪律、验证门
  references/
    deployment-pattern.md           # 详细模式:如何为新模型实例化模板、契约细节、渲染器指南、纪律全表、端到端验证清单
    templates/
      backend_server.py             # 通用 stdlib server + ModelManager(模型无关;输入解码/输出装配为 seam)
      runner_adapter.py             # Runner 协议 + _import_from + build_runner 工厂 + 1 个填空示例
      frontend_app.py               # 通用 gradio 骨架(输出渲染为可插拔 render_output)
      config.yaml                   # 配置模式样例
      run_backend.sh                # 后端容器模板(MODEL 参数化;缓存从 REPO_ROOT/caches 挂载)
      run_frontend.sh               # 前端模板(0.0.0.0 + 端口可覆盖 + 代理处理)
      stop_all.sh                   # 停后端容器
```

## 4. 纪律(从 phase1 承接 + 本次实测踩坑提炼为通用条目)

- **Cache 落项目目录**:模型权重下载到 `<repo>/caches/{modelscope,huggingface,torch}`,`:ro` 挂载到容器,env 注入 `HF_HOME`/`MODELSCOPE_CACHE`/`TORCH_HOME`,**绝不依赖 `~/.cache`**。(本次 layerd 后端误挂 `~/.cache` 导致离线加载失败,已提炼为模板纪律。)
- **容器**:`--runtime=nvidia`(否则 torch.cuda 失效)、`--user $UID:$GID`(否则产物 root 所有)、按空闲显存选卡(复用 `gpu_select.py`)。
- **代理**:容器内 unset 全套 `*_proxy`;host 本地 curl 的 `NO_PROXY` 必须含 `127.0.0.1`(不只 `localhost`),否则被 SOCKS/tinyproxy 劫持 500。
- **共享机端口**:默认端口可能被同机别的租户占用,先 `ss -ltn` 探活再选空闲口,前端用 `PORT=` 覆盖。
- **UI 参数须匹配模型真实取值域**:前端控件只暴露模型实际接受的值(本次 qwen `resolution` 用 Slider 放进了模型只接受 640/1024 之外的值,直接 500;改 Radio)。提炼为通用校验条目。
- **端到端冒烟门**:部署后必须 `/health` → `/predict` 一个小输入 → 断言响应契约,两端都过才算完成。

## 5. 配套 git

- `web/` 已 `git rm --cached` 并加入 `.gitignore`(像 `runs/`、`caches/`),文件留磁盘。
- 新 skill 是被跟踪产物,提交进仓库。
