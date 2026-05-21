# autoexplore 第一版设计 · 第一阶段(baseline 模型自动复现)

**日期**: 2026-05-19
**范围**: 需求文档第一阶段步骤 1–10 端到端
**形态**: Claude Code Skill 编排 + Python 工具脚本(混合)

## 1. 目标与范围

构建第一版「开源模型自动复现」agent,覆盖[需求文档](../../../需求文档-20260521.md)**第一阶段**全部 10 步:从需求澄清、建立测试集、设计指标,到搜索/排序开源模型、Docker 复现循环、测试集推理、计算指标、选出最佳模型。

**核心分工**:Skill(SKILL.md)负责*判断与编排*(需求澄清、设计测试集与指标、搜索/排序模型、调试失败);Python 脚本负责*确定性重活*(选卡、Docker 操作、跑推理、算指标、读写进度)。两者通过**文件系统约定**(工作目录 + JSON schema)通信,而非函数调用。

### 第一版明确不做(避免 scope 膨胀)
- 第二阶段(效果迭代优化)——独立的下一个 spec。
- 多机/集群调度——只在单台 8 卡服务器内选卡。
- 训练/微调——只复现 + 推理 + 评测。
- 密钥管理(HF token 等)——要求用户预先在环境配好。
- 并行复现多模型——第一版串行,一次一个。

## 2. 运行环境假设

- 云 GPU 服务器,已装 Docker + NVIDIA runtime。
- 8 张 GPU 可用;**按显卡空闲度自动决定用哪几张**。
- GPU 选择是动态的:每次进入复现循环的环境准备阶段时查 `nvidia-smi`,挑当前空闲卡。

## 3. 总体架构与目录结构

```
autoexplore/
├── SKILL.md                      # 流程与决策规则(Claude 读它执行)
├── scripts/
│   ├── gpu_select.py             # 查 nvidia-smi,按空闲度挑卡 → 打印 CUDA_VISIBLE_DEVICES
│   ├── docker_env.py             # 构建/复用镜像、在容器内执行命令
│   ├── run_inference.py          # 在容器里对测试集跑推理 → predictions
│   ├── compute_metrics.py        # 调用方向专属 eval 脚本 → metrics.json
│   └── progress.py               # 读写 progress.json / results.tsv 的 CLI
├── references/
│   └── reproduction-loop.md      # 复现循环细则(从 SKILL.md 引用,避免主文件过长)
└── runs/
    └── <run-tag>/                # 一次研究运行(如 vqa-mar5)
        ├── plan.md               # 需求澄清结论 + 测试集/指标设计(人工确认过)
        ├── dataset/              # 统一测试集(不可变)
        ├── evaluate.py           # 方向专属指标计算(不可变,ground truth)
        ├── candidates.json       # 候选模型 + 优先级排序
        ├── results.tsv           # 各模型最终指标汇总
        └── models/
            └── <model-name>/     # 每个模型一个工作目录
                ├── progress.json # 复现状态:阶段/重试次数/最后错误
                ├── run.log       # 运行输出(隔离,不污染上下文)
                ├── Dockerfile    # 该模型的环境
                └── predictions.jsonl  # 推理输出(每行一条 JSON)
```

### 关键设计点
- `dataset/` 与 `evaluate.py` 一旦人工确认就**不可变**——保证所有模型可比(对齐 autoresearch 示例最重要的原则)。
- 每个模型隔离在 `models/<name>/`,互不干扰,可独立重试/恢复。
- 脚本输出走 `run.log`,Claude 只 `grep` 关键行,不让训练/推理日志淹没上下文。

## 4. 端到端数据流(10 步映射)

**J** = Claude 判断,**S** = 脚本执行,**H** = 人工确认。

| # | 步骤 | 角色 | 动作与产出 |
|---|------|------|-----------|
| 1 | 需求分析与方向澄清 | J + H | 对话明确方向;商定 run-tag → 创建 `runs/<tag>/`,写 `plan.md` 草稿 |
| 2 | 建立统一测试集 | J + H | 按方向提议数据来源与规模,人工确认 → 落到 `dataset/`(此后不可变) |
| 3 | 设计评价指标 | J + H | 写 `evaluate.py`(输入 predictions + dataset,输出 `metrics.json`),人工确认后冻结 |
| 4 | 寻找开源模型 | J | 检索(HF / paperswithcode / arxiv),收集候选 |
| 5 | 优先级排序,选 ≤3 | J | 按官方披露效果排序 → 写 `candidates.json`(name, repo, reported_score, priority) |
| 6–7 | 逐个复现(复现循环) | S + J | 对每个候选进入复现循环(见第 5 节),产出可运行容器环境 |
| 8 | 测试集推理 | S | `run_inference.py` 在容器内对 `dataset/` 跑推理 → `predictions.jsonl` |
| 9 | 计算指标 | S | `compute_metrics.py` 调 `evaluate.py` → `models/<name>/metrics.json` |
| 10 | 选最佳模型 | S + J | `progress.py` 汇总各模型指标到 `results.tsv`;Claude 据此选出最佳 |

### 数据流约定
- **plan.md 是人机契约**:步骤 1–3 结论写入,人工确认后才进入步骤 4。这是唯一的"暂停问人"关卡;之后 4–10 尽量自主。
- **predictions 中间格式**:`plan.md` 约定"每行一条 JSON",`run_inference.py` 产出该格式,`evaluate.py` 负责解读 → 推理与指标脚本解耦。
- **metrics.json 统一接口**:无论方向,归一成 `{"primary_metric": <float>, "metrics": {...}}`;`results.tsv` 只读 `primary_metric` 排序。
- **失败也写进 results**:某模型复现/评测失败,记 `status=crash`,不阻塞其他模型。

## 5. 复现循环细则(步骤 6–7 核心)

细则写进 `references/reproduction-loop.md`,SKILL.md 引用它。对**单个候选模型**:

```
进入 models/<name>/,读 progress.json(若存在则从中断处恢复)

阶段 A: 准备环境
  1. gpu_select.py → 查 nvidia-smi,按显存空闲度挑卡,返回如 "2,5,7"
  2. docker_env.py build → 若镜像已存在则跳过;否则按 Dockerfile 构建
     (Dockerfile 初版由 Claude 参照模型仓库 README/requirements 生成)

阶段 B: 下载
  3. docker_env.py run "<下载命令>" → 容器内拉安装包 + 模型权重(输出 → run.log)

阶段 C: 验证推理
  4. 有官方示例代码 → 跑示例;没有 → Claude 自建最小推理脚本
  5. docker_env.py run "<推理验证命令>" > run.log
  6. grep 成功标志 / 检查退出码

  成功 → 循环结束,progress.json 标 status=ready
  失败 → 进入阶段 D
```

### 阶段 D:有界重试(对齐 autoresearch 纪律)
```
  retry_count += 1(写入 progress.json)
  Claude 读 `tail -n 50 run.log` 的栈/错误,判断:
    - 简单可修(缺依赖/路径/CUDA 版本) → 调整 Dockerfile 或命令 → 回阶段 A
    - 根本性失败(架构不兼容/资源不够) → 放弃
  若 retry_count >= 3 → 放弃该模型,progress.json 标 status=crash,进入下一候选
```

### progress.json schema(支持中断恢复)
```json
{
  "model": "llava-1.5",
  "stage": "C",                    // A/B/C/ready/crash
  "retry_count": 1,
  "gpus": "2,5,7",
  "image_tag": "autoexplore/llava-1.5",
  "last_error": "ImportError: flash_attn ...",
  "updated_at": "2026-05-19T..."
}
```

### 明确决定
- **重试上限 3 次**(对齐 CLAUDE.md "3 次后停下重新评估"原则),用完即放弃该模型,不卡死链路。
- **GPU 选卡每次进入阶段 A 动态查**——别的进程可能占卡;脚本只返回建议,通过 `CUDA_VISIBLE_DEVICES` 传进容器。
- **镜像可复用**:`docker_env.py build` 先查同名镜像存在则跳过(对齐需求文档"已构建过则忽略")。
- **日志隔离**:容器输出进 `run.log`,Claude 只在失败时 `tail`。

## 6. 错误处理与边界

| 失败类型 | 处理层 | 策略 |
|---------|-------|------|
| GPU 全忙/无空闲卡 | `gpu_select.py` | 非零退出码 + 明确信息;Claude 决定等待还是用更少卡 |
| 无 NVIDIA runtime / Docker 不可用 | `docker_env.py` | 启动即 fail fast,打印诊断(`docker info` 摘要),不进入循环 |
| 镜像构建失败 | 阶段 A | 计入 retry_count,Claude 读日志改 Dockerfile |
| 下载失败(网络/权限/token) | 阶段 B | 区分可重试(网络)与不可重试(权限);后者直接放弃并标因 |
| 推理崩溃(OOM/bug) | 阶段 C→D | 有界重试 3 次 |
| `evaluate.py` 报错 | `compute_metrics.py` | 该模型评测失败,`status=crash`,不影响其他模型 |
| 整个 run 中断 | `progress.py` | 下次从 `progress.json` 的 `stage` 恢复,已 ready 模型跳过 |

**fail-fast 原则**:环境前置条件(Docker、NVIDIA runtime、GPU)在进入循环前一次性检查,缺了带诊断信息退出。

## 7. 测试策略

Python 标准库 + pytest。脚本可独立测试;Claude 判断部分靠 `plan.md` 人工确认关卡兜底。

### 单元测试(不碰真实 Docker/GPU)
| 脚本 | 测什么 | 怎么测 |
|------|--------|--------|
| `gpu_select.py` | 解析 nvidia-smi、按空闲度选卡 | 喂入捕获的 nvidia-smi 文本 fixture,断言选出卡 ID;含"全忙"边界 |
| `progress.py` | 读写 progress.json、汇总 results.tsv、中断恢复 | tmp 目录建假 run,断言 schema 字段、恢复时跳过 ready 模型 |
| `compute_metrics.py` | 调 evaluate.py、归一成 metrics.json | 玩具 evaluate.py + 假 predictions,断言 `primary_metric` 提取 |
| `docker_env.py` | 镜像存在判断、命令拼装、退出码处理 | mock `subprocess`,断言 docker CLI 调用参数;不真起容器 |

### 契约测试(脚本间文件约定)
- 一个端到端 *dry-run*:玩具 dataset + 玩具 evaluate.py,模拟整条 1→10 的**文件流转**(plan.md → candidates.json → predictions → metrics.json → results.tsv),断言每步产物 schema 对得上。Docker/推理用 stub。这是最有价值的测试——锁住脚本间接口。

### 冒烟测试(需真实环境,默认 skip)
- 极小已知模型在真实 Docker 跑通阶段 A→C,标 `@pytest.mark.gpu`,只在有 GPU 服务器手动跑。

### 不测
- Claude 判断质量(搜模型、设计指标)——靠 `plan.md` 人工确认。
- 真实模型复现成功率——agent 运行结果,非单元测试范畴。

**TDD 节奏**:每个脚本先写测试(红)→ 实现(绿)→ 重构,小步提交。
