# Web 部署模式详解

把 phase1/phase2 产出的某个 `runs/<tag>/` 下的 `infer.py`，部署成「host 上的 Gradio 前端 + 每模型一个后端容器」。本文给出：如何为一个新模型实例化模板、请求/响应契约细节、输出渲染器指南、纪律全表、端到端验证清单。

**前提**:`templates/` 是骨架，不要求 import 即跑。落地时把模板拷进部署仓库的 `web/` 包，按下述 seam 填空。`web/` 本身像 `runs/`、`caches/` 一样**不进 git**(`.gitignore` + `git rm --cached`),只有 skill 进仓库。

## 1. 目录落地形态

模板拷进部署仓库后，典型 `web/` 包长这样（模板里的单文件在落地时按职责拆进子模块）:

```
web/
  config.yaml                 # 由 templates/config.yaml 填空而来
  common/
    config.py                 # load_config + merge_params（见下方“通用 common 层”）
    <modality>.py             # 输入解码 / 输出装配的模态实现（如 imaging.py、text.py）
  backend/
    server.py                 # = templates/backend_server.py，填好两个 SEAM
    runners.py                # = templates/runner_adapter.py，每模型一个 Runner + build_runner
  frontend/
    app.py / ui.py            # = templates/frontend_app.py，填好 SEAM A/B
    client.py                 # HTTP 客户端（见下方“通用 client”）
  scripts/
    run_backend_<model>.sh    # 一模型一份，由 templates/run_backend.sh 填空
    run_frontend.sh
    stop_all.sh
  tests/                      # 见“端到端验证清单”
```

后端容器内 `PYTHONPATH=/app`、`web/` 挂到 `/app/web`，故都以 `python -m web.backend.server` 跑、import 写 `from web.common...`。模板文件里写的是相对 import(`from config import ...`)，落地时按你的包布局改成 `from web.common.config import ...`。

## 2. 五个 seam（部署时唯一要写的东西）

| seam | 位置 | 填什么 |
|---|---|---|
| **build_runner / Runner** | `backend/runners.py` | `_import_from` 复用 infer.py 的 `load_*`/推理入口，适配成 `load/infer/unload/health_meta`。**复用，不重写推理。** |
| **SEAM 1 输入解码** | `backend/server.py` `decode_request` | 把 `req[<input-key>_b64]` 解成 infer 接受的对象（图像→PIL、文本→str…） |
| **SEAM 2 输出装配** | `backend/server.py` `assemble_response` | 把 `runner.infer()` 原始输出装成 JSON 可序列化的响应 dict |
| **SEAM A 参数收集** | `frontend/app.py` `collect_params` | UI 控件值 → params dict（键与模型一致） |
| **SEAM B 输出渲染** | `frontend/app.py` `render_output` | 响应 → UI 组件（HTML / Image / JSON / 交互画布…） |

`_INFER_KEYS`(在 `common/config.py`)也算半个 seam:列出每个模型 `infer()` 真正吃的参数键，`merge_params` 只放行这些键,挡掉前端误传的非法键。

## 3. 请求/响应契约

- **请求**:`{"<input-key>_b64": <base64>, "params": {...}}`。`<input-key>` 随模态命名(图像用 `image_b64`，文本可用 `text` 直接传不编码)。`params` 经 `merge_params` 裁剪。
- **响应(成功 200)**:`{"model": <name>, "infer_ms": <int>, <模态相关输出>, ...}`。输出形状随模态:
  - 图像:`{"image_b64": ...}` 或多图 `{"layers": [b64, ...]}`。
  - 文本:`{"text": ...}`。
  - 结构化:任意 JSON。
- **响应(错误 500)**:`{"error": repr(e), "hint": <可选排障提示>}`。server 模板已兜底:OOM 时给「换更小参数」提示,异常不崩连接。
- **/health(GET)**:`{"loaded": bool, "model": name, "free_mib": ..., <health_meta>}`。前端可据此显示模型是否已驻显存。

## 4. 通用 common 层（直接抄，几乎不改）

**`common/config.py`** — `load_config(path)` 读 yaml 与内置 `DEFAULTS` 深合并；`merge_params(model, user, cfg)` 按 `_INFER_KEYS[model]` 裁剪参数。结构见 worked example,把 `DEFAULTS`/`_INFER_KEYS` 换成你的模型即可。

**`frontend/client.py`** — 极薄 HTTP 客户端:
```python
import requests
def predict(base_url, model_input, params, timeout=600):
    r = requests.post(f"{base_url}/predict",
                      json={"<input-key>_b64": encode(model_input), "params": params or {}},
                      timeout=timeout)
    r.raise_for_status(); return r.json()
def health(base_url, timeout=5):
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout); r.raise_for_status()
        return {**r.json(), "online": True}
    except Exception:
        return {"loaded": False, "online": False}
```

## 5. 输出渲染器指南（SEAM B）

按输出模态选渲染方式，与 `out` 组件类型对齐:

- **图像**:`assemble_response` 回 `image_b64`/`layers`;`render_output` 解码成 `np.ndarray`/`PIL`,`out = gr.Image()` 或 `gr.Gallery()`。
- **文本**:回 `text`;`out = gr.Markdown()`/`gr.Textbox()`。
- **结构化 JSON**:`out = gr.JSON()`。
- **交互组件(如可拖拽画布、可视化编辑器)**:`render_output` 产出一段 HTML(内联数据),`out = gr.HTML()`,前端脚本经 `launch(head=YOUR_JS)` 注入。
  - gradio v6:自定义 `<head>` JS 走 `launch(head=...)`,**不是** `gr.Blocks(head=...)`(v6 的 Blocks 已无 head 参数)。
  - 这类复杂交互(本 worked example 的图像分层画布即属此类)是**领域特例,绝不进 skill 模板**——模板只在此处举例说明「一种渲染器」。

## 6. 纪律全表（phase1 承接 + web 实测踩坑）

| 类别 | 规则 |
|---|---|
| **Cache 落项目目录** | 权重下载到 `<repo>/caches/{modelscope,huggingface,torch}`,`:ro` 挂载到容器,env 注入 `HF_HOME`/`MODELSCOPE_CACHE`/`TORCH_HOME`。**绝不依赖 `~/.cache`**(`HOME=/tmp` 强制隔离)。worked example 曾误挂 `~/.cache` 导致离线加载失败。 |
| **容器** | `--runtime=nvidia`(否则 torch.cuda 失效);`--user $UID:$GID`(否则产物 root 所有);按空闲显存选卡(复用 phase1 `gpu_select.py`)。 |
| **离线加载** | 容器内 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`;`trust_remote_code` 模型需 `HF_MODULES_CACHE=/tmp/hf_modules`(可写)。 |
| **代理** | 容器内 unset 全套 `*_proxy`;host curl 的 `NO_PROXY` 必须含 `127.0.0.1`(不只 `localhost`),否则本机后端被 SOCKS/tinyproxy 劫持 500。 |
| **端口** | 后端绑 `127.0.0.1:<port>`(外部不可达);前端绑 `0.0.0.0:<port>`。共享机默认口(7860)可能被别的租户占,先 `ss -ltn` 探活再选空闲口,`PORT=` 覆盖。 |
| **idle_ttl** | 必须大于最坏单次推理耗时,否则长推理被 idle 线程中途卸载。 |
| **UI 取值域** | 前端控件只暴露模型实际接受的值:离散值(如 resolution 仅 640/1024)用 `Radio`/`Dropdown`,别用 `Slider` 放进非法值(worked example 曾因此 500)。`_INFER_KEYS` + `merge_params` 是第二道防线。 |
| **重模型镜像不塞 web 框架** | gradio/requests 装在 host `.venv`(前端);后端容器只 stdlib `http.server`,镜像复用 phase1 的复现镜像不变胖。 |

## 7. 端到端验证清单（两端都过才算完成）

1. **单测**(host `.venv`,纯逻辑,不碰 GPU):`merge_params` 裁剪正确、输入编解码可逆、`assemble_response` 形状对、`ModelManager` 懒加载/卸载逻辑、client 拼 URL/JSON 正确。
2. **后端冒烟**:`bash scripts/run_backend_<model>.sh` → `docker logs` 看到 `[backend] model=... port=...` → `curl --noproxy 127.0.0.1 127.0.0.1:<port>/health` 返回 `{"loaded": false, ...}`。
3. **/predict 冒烟**:`curl --noproxy 127.0.0.1 -X POST 127.0.0.1:<port>/predict -d '<小输入 json>'` → 断言响应含 `model`/`infer_ms` 与模态输出键;首次会触发懒加载(慢),`/health` 随后 `loaded:true`。
4. **前端冒烟**:`bash scripts/run_frontend.sh` → VSCode 转发端口 → 选模型 → 传小输入 → 跑通 → 渲染正常 → (可选)下载。
5. **idle 卸载**:静置 > `idle_ttl` 后 `/health` 应回到 `loaded:false`(显存释放)。

任一端不过都不算部署完成。冒烟脚本可固化进 `tests/test_backend_smoke.py`,标 GPU 用例,CI 缺卡时跳过。
