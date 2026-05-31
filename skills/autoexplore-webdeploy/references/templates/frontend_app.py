"""通用 Gradio 前端模板:选模型 -> 传输入 -> 调后端 -> 渲染输出 -> 可选下载。
绑定 0.0.0.0:<port>，靠 VSCode 端口转发访问。

**与模态无关。** 输入控件与 render_output 是 seam，按你的输入/输出模态填。
"""
from __future__ import annotations

import os

import gradio as gr

from config import load_config
import client  # 部署提供的后端 HTTP 客户端（见 deployment-pattern.md）

CFG = load_config(os.environ.get("CONFIG_PATH"))
MODELS = list(CFG["models"].keys())


def _base_url(model: str) -> str:
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")   # 后端绑 127.0.0.1
    return f"http://{host}:{CFG['ports'][model]}"


# ── SEAM A:把 UI 控件值打包成后端 params（按模型裁剪）。
def collect_params(model: str, *widget_values) -> dict:
    raise NotImplementedError("SEAM A: 按模型把控件值组装成 params dict")


# ── SEAM B:把后端响应渲染成 UI 输出（HTML / Image / JSON / 交互组件…）。
#    复杂交互（如可拖拽画布）在这里产出 HTML，并配合 head=<JS> 注入前端脚本。
def render_output(resp: dict):
    raise NotImplementedError("SEAM B: 按模态把 resp 渲染成 UI 输出")


def run_infer(model, raw_input, *param_widgets):
    if raw_input is None:
        raise gr.Error("请先提供输入")
    params = collect_params(model, *param_widgets)            # SEAM A
    try:
        resp = client.predict(_base_url(model), raw_input, params, timeout=600)
    except Exception as e:
        raise gr.Error(f"后端推理失败：{e}")
    return render_output(resp), resp                          # 第二个返回值存进 gr.State 供下载等复用


def build() -> gr.Blocks:
    with gr.Blocks(title="模型部署") as demo:
        gr.Markdown("## 模型部署\n选择模型 → 提供输入 → 运行。")
        with gr.Row():
            model = gr.Dropdown(MODELS, value=MODELS[0], label="模型")
            raw_input = gr.Image(label="输入", type="numpy")  # SEAM:换成你的输入控件
        with gr.Accordion("高级参数", open=False):
            # SEAM:每个模型一组参数控件。**控件取值域必须匹配模型真实接受范围**——
            # 只接受离散值时用 Radio/Dropdown，别用 Slider 放进非法值（否则后端 500）。
            param_a = gr.Slider(1, 10, value=4, step=1, label="示例参数")
        run = gr.Button("运行", variant="primary")
        out = gr.HTML()           # SEAM:换成与 render_output 返回类型匹配的输出组件
        state = gr.State()

        run.click(run_infer, [model, raw_input, param_a], [out, state])
    return demo


def main():
    port = int(os.environ.get("PORT", CFG["ports"]["frontend"]))
    # gradio v6：自定义 <head> JS（如交互渲染器的脚本）走 launch(head=...)，
    # 不是 gr.Blocks(head=...)（v6 的 Blocks 已无 head 参数）。
    build().launch(server_name="0.0.0.0", server_port=port, share=False)
    #                                                        , head=YOUR_JS)  # 有交互渲染器时加


if __name__ == "__main__":
    main()
