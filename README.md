<div align="center">

# autoexplore

**面向研究方向的自主开源模型复现与迭代优化智能体**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](pyproject.toml)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-D97757.svg)](https://claude.com/claude-code)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#项目状态)

简体中文 | [English](README.en.md)

</div>

---

## 简介

**autoexplore** 是一套以 [Claude Code](https://claude.com/claude-code) **Skill** 形式交付的自主智能体。你只需给出一段**研究方向描述**，它就能自动完成「找模型 → Docker 里复现 → 在统一测试集上评测 → 选出最优 baseline → 不断迭代优化」的全流程，并可把成果一键部署成可交互的 Web 应用。

它不是一个传统应用，而是一组**任务/模型无关的可复用技能**——同样的流程既能跑视觉问答，也能跑分割、生成或任意有公开权重的开源模型方向。

## 核心能力

- 🔭 **方向澄清 → 指标冻结**：与你对齐研究方向，建立**不可变**的统一测试集与评价指标，保证所有模型/实验横向可比。
- 🤖 **自动复现 baseline**：在 Hugging Face / paperswithcode / arxiv 检索候选模型并排序，逐个在 Docker 里复现（建/复用镜像、下权重、验证推理、有界重试），跑测试集算分，选出最优。
- ♾️ **不终止迭代优化**：诊断模型短板 → 在可信数据源（Arxiv / Hugging Face / paperswithcode）搜索改进方向 → 并行实现与训练 → 评分 → 相对提升 +5% 才晋升新主干，循环直到人工终止。
- 🌐 **一键 Web 部署**：复用复现产物的 `infer.py`，拉起「Gradio 前端 + 每模型一个后端容器」的可交互应用，模型懒加载、空闲自动卸载。
- 🧰 **确定性脚本骨架**：选卡、Docker 编排、推理、算分、状态机等关键环节由 `scripts/` 下经过测试的 Python 脚本承担，智能体只负责决策，可中断、可恢复。

## 工作流概览

```
研究方向描述
     │
     ▼
┌─────────────────────────── 第一阶段:Baseline 筛选 ───────────────────────────┐
│ 1. 澄清方向 → 2. 建测试集(冻结) → 3. 设计指标(冻结) →                         │
│ 4. 搜索候选模型 → 5. 排序选 ≤3 个 →                                            │
│ 6-9. 逐个 Docker 复现 + 推理 + 评分 → 10. 选指标最高者为 baseline             │
└──────────────────────────────────────────────────────────────────────────────┘
     │  baseline 模型
     ▼
┌──────────────────────── 第二阶段:迭代优化(不终止) ─────────────────────────┐
│ 诊断短板 → 推理管道调优闸门(便宜档先行) →                                    │
│ LOOP: 搜 3 个方向 → 实现/训练 → 评分 → 相对 +5% 则晋升新主干 → 重新诊断       │
│ (绝不停下问人,直到人工终止)                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
     │  最优模型 + infer.py
     ▼
   Web 部署(Gradio 前端 + 每模型后端容器)
```

## 运行环境

autoexplore 面向**云端多 GPU 服务器**：

| 依赖 | 说明 |
|------|------|
| Claude Code | 智能体的运行宿主，技能由其加载与调度 |
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | 依赖与脚本运行器（推荐） |
| Docker | 模型复现/部署的隔离环境 |
| NVIDIA Container Runtime | 容器内访问 GPU（`--runtime=nvidia`） |
| GPU | 多卡（参考配置 8 卡），按空闲显存自动选卡 |

## 安装

```bash
# 1. 克隆仓库
git clone <仓库地址> autoexplore
cd autoexplore

# 2. 安装依赖(脚本运行环境)
uv sync

# 3. 在 Claude Code 中加载技能
#    skills/ 下的三个技能即开即用,在 Claude Code 会话中按需调用。
```

## 使用

在 Claude Code 会话中，描述你的研究方向，按阶段调用对应技能：

| 技能 | 何时使用 |
|------|---------|
| **`autoexplore-phase1`** | 针对某个研究方向，复现并评测 ≤3 个开源模型，选出指标最高的 baseline。 |
| **`autoexplore-phase2`** | 在 phase-1 产出的 baseline 上做不终止的迭代优化，每次相对提升 +5% 晋升新主干。 |
| **`autoexplore-webdeploy`** | 把复现/优化得到的 `infer.py` 部署成可交互 Web 应用。 |

> 第一/二阶段各只有一个**人工关卡**（确认方向、测试集与指标）；确认后智能体即自主运行，不再逐步问人。每个 run 的工作目录落在 `runs/<tag>/`（默认不纳入版本管理）。

### 确定性脚本速查

智能体依赖一组经过测试的 Python 脚本完成关键的确定性步骤：

| 脚本 | 用途 |
|------|------|
| `scripts/gpu_select.py` | 按空闲显存选卡，打印 `CUDA_VISIBLE_DEVICES` |
| `scripts/docker_env.py` | Docker 检查 / 镜像构建复用 / 容器执行 |
| `scripts/run_inference.py` | 容器内推理 → `predictions.jsonl` |
| `scripts/compute_metrics.py` | 调用冻结的 `evaluate.py` → `metrics.json` |
| `scripts/progress.py` | 第一阶段进度持久化与结果汇总 |
| `scripts/phase2_state.py` | 第二阶段状态机 / 晋升门 / 实验派发 |
| `scripts/diagnose.py` | 黑盒调用 `evaluate.py` 做短板分解 |
| `scripts/directions_schema.py` | 优化方向 JSON schema 校验 |
| `scripts/train_launch.py` | 数据出处 / 多卡训练启动 / ckpt 续训 |

## 设计纪律

这些约束是结果可信、流程可恢复的根基：

- **评测即真相，且不可变**：测试集 `dataset/` 与评价脚本 `evaluate.py` 一经人工确认即冻结，保证所有模型与实验横向可比。
- **容器纪律**：每次 `docker run` 都带 `--user $UID:$GID --runtime=nvidia`；模型权重 / 数据落在仓库内 `caches/{modelscope,huggingface,torch}` 并以 `:ro` 挂载，绝不依赖 `~/.cache`。
- **有界重试**：每个模型/实验重试上限 3 次，失败标 `crash` 但不阻塞同批其他任务。
- **可中断、可恢复**：入口先读进度/状态文件，跳过已完成项，已晋升主干不回退。
- **晋升只认主指标相对 +5%**：避免随机噪声造成主干抖动。
- **日志不污染上下文**：容器/训练输出重定向到日志文件，仅在失败时 `tail`。

## 仓库结构

```
autoexplore/
├── skills/                       # 三个 Claude Code 技能(核心交付物)
│   ├── autoexplore-phase1/       #   第一阶段:复现 baseline
│   ├── autoexplore-phase2/       #   第二阶段:迭代优化
│   └── autoexplore-webdeploy/    #   Web 部署
├── scripts/                      # 确定性 Python 脚本(选卡/Docker/推理/算分/状态机)
├── tests/                        # pytest 测试(GPU 用例默认跳过)
├── docs/                         # 设计规格(specs)与实现计划(plans)
├── examples/                     # autoresearch 参考设计(自主实验循环的范式)
├── 需求文档-20260521.md           # 原始需求文档
├── CLAUDE.md                     # 给 Claude Code 的项目指引
├── GIT_CONVENTIONS.md            # Git 分支与提交规范
├── pyproject.toml
└── LICENSE
```

## 开发

```bash
# 安装含开发依赖
uv sync

# 运行测试(默认跳过需要真实 Docker + GPU 的用例)
uv run pytest

# 在具备 Docker + GPU 的机器上运行全部用例
uv run pytest -m gpu
```

## 贡献

欢迎贡献！请遵循 [GIT_CONVENTIONS.md](GIT_CONVENTIONS.md)：

- **分支**：`feature/*`、`bugfix/*`（基于 `develop`），`hotfix/*`（基于 `main`）。
- **提交**：[Conventional Commits](https://www.conventionalcommits.org/) —— `<type>(<scope>): <subject>`。
- **合并**：`feature` / `bugfix` 走 squash-and-merge；已推送的提交**不要 rebase**，用 merge 集成。

## 项目状态

active —— 三个技能与确定性脚本骨架均已落地并有测试覆盖；核心流程持续演进中。

## 许可证

本项目以 [Apache License 2.0](LICENSE) 开源。

## 致谢

- 基于 [Claude Code](https://claude.com/claude-code) 的 Skill 机制构建。
- 自主实验循环的范式参考了 [`examples/autoresearch_program.md`](examples/autoresearch_program.md)。
