---
name: autoexplore-phase2
description: Use after phase-1 produced a baseline — drives phase-2 iterative optimization: set the chosen model as the backbone, diagnose its test-set weaknesses, try cheap inference-pipeline tuning first, then run a never-ending search loop (Arxiv/HF/paperswithcode → 3 directions → train if needed → score → promote on relative +5%).
---

# autoexplore 第二阶段:模型效果迭代优化

输入:第一阶段在 `runs/<tag>/` 产出的冻结 `dataset/`、`evaluate.py`、`results.tsv` 与 ≥1 个 `status=ready` 模型。
产出:在同方向上不终止地优化"模型主干",每次有明显提升(相对 +5%)就晋升新主干,直到人工终止。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡;**复用第一阶段主干的 docker 镜像**;按空闲度选卡。

**这是通用 skill,不绑定任何具体任务/数据集/模型。** 只依赖第一阶段冻结的文件契约
(`metrics.json` 的 `{primary_metric, metrics{}}`、`evaluate.py` 的 CLI、`results.tsv` 列),不依赖任务语义。

## 流程(需求文档第二阶段)

唯一真相源 = `runs/<tag>/phase2/state.json`,只经 `scripts/phase2_state.py` 读写。任意时刻先 `resume` 决定续点:
```bash
uv run scripts/phase2_state.py resume --run-dir runs/<tag>
```

### 步骤 0:Setup(唯一人工关卡)
续用第一阶段 `<tag>`;`git checkout -b optimize/<tag>`;`phase2_state init` 设主干 = ready 最高分;
校验 `dataset/`+`evaluate.py` 仍冻结。确认后进入自主循环,之后不再逐步问人。

### 步骤 1:短板诊断(每次主干变化重跑)
`diagnose.py` 把冻结 `evaluate.py` 当黑盒在子集视图上跑 → `diagnostics/diag_<id>.{json,md}`;
读 `.md` 写短板小结(最差分组/样本/副指标)。

### 步骤 2:推理管道调优闸门(便宜档先行)
判断"仅调推理管道(参数/后处理,不训练)"是否有改善空间;有就先跑 `infer-tune` 实验,
过门即晋升回步骤 1;无空间则 `inference_tuning=explored` 进步骤 3。

### 步骤 3:不终止搜索循环(复用主干镜像)
LOOP:搜 Arxiv/HF/paperswithcode 选 3 方向(`directions.json`,`directions_schema.py` 校验、跨轮去重)→
`dispatch` 按空闲卡并发、训练型排队 → 各 `exp_*/` 按 tier 实现(需训练用 `train_launch.py`)→
`run_inference`+`compute_metrics` 评分 → `gate` 判相对 +5% → 过门 `promote_backbone`+commit 回步骤 1,
否则记 `discard/crash` 继续选新方向。**绝不停下问人**,直到人工终止。

完整细则见 [references/optimization-loop.md](references/optimization-loop.md)。

## 关键纪律
- `dataset/` 与 `evaluate.py` 全程**不可变**,保证主干各版本与实验可比。
- **便宜档先行**:推理管道调优闸门先于昂贵搜索;搜索内 config/post-process 优先于训练。
- **晋升只认主指标相对 +5%**,避免噪声/随机种子的微小波动误晋升致主干抖动。
- **keep/discard 用 state.json 主干指针,不用 git reset**:失败实验目录保留作研究档案。
- 容器/训练输出进 log,只失败时 `tail`;逐实验重试上限 3,crash 不阻塞同轮兄弟。
- 中断可恢复:入口 `resume` 读 state.json;已 scored slot 跳过、已晋升主干不回退。
- **容器纪律(继承第一阶段)**:每次 `docker run` 带 `--user $UID:$GID --runtime=nvidia`;
  caches 以 `:ro` 挂 `/cache/{modelscope,huggingface,torch}`,env 注入对应 `*_CACHE`/`*_HOME`。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/phase2_state.py {init,resume,gate,dispatch,backbone-get}` | 状态/晋升门/派发(确定性核心) |
| `scripts/diagnose.py` | 黑盒调用 evaluate.py 出短板分解 |
| `scripts/directions_schema.py --file <directions.json>` | 方向 schema 校验 |
| `scripts/train_launch.py` | 数据出处/多卡启动/ckpt 续/预算 |
| `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py` | 复用第一阶段 |
