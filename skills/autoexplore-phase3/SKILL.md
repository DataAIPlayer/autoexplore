---
name: autoexplore-phase3
description: Use after phase-2 produced a winning backbone — drives phase-3 inference-speed optimization to production: 3a pick the fastest quality-passing single-card acceleration framework, 3b run a dual-gated single-card scheme loop (search GitHub/Arxiv/HF/paperswithcode → 3 directions → benchmark → accept on latency −10% AND quality loss ≤1%, compounding onto the base), then auto-advance on saturation to 3c multi-card parallelism, picking the fastest quality-passing scheme as the final production solution.
---

# autoexplore 第三阶段:模型推理速度优化

输入:第二阶段在 `runs/<tag>/phase2/state.json` 产出的获胜 `backbone`(权重 + 质量分);第一阶段冻结的 `dataset/`、`evaluate.py` 仍在位。
产出:生产级最终多卡推理方案(`phase3/state.json` 的 `final`),全程质量损失 ≤1%。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡;**复用主干 docker 环境**;按空闲度选卡。

**这是通用 skill,不绑定任何具体任务/数据集/模型。** 只依赖前阶段冻结的文件契约
(`evaluate.py` 的 CLI、`phase2/state.json` 的 backbone、`metrics.json` 的 `{primary_metric, metrics{}}`),不依赖任务语义。

## 两个冻结地基(全程不可变,保证可比)

- 质量:沿用 `evaluate.py` + `baseline/metrics.json`(= phase2 主干分)。
- 速度:`scripts/benchmark.py`(冻结测速协议)+ `phase3/bench_config.json`(本 run 钉死:固定子集/warmup/iters/卡型)+ `baseline/speed.json`。
- **唯一可变 = `adapter.py`**:Claude 按每个框架/方案写,暴露 `load_model()` / `infer_one(record)`。`benchmark.py` 一次执行同产 `speed.json` 与 `predictions.jsonl` → 喂 `evaluate.py` 出 `metrics.json`。同一次执行同时测速测质,杜绝漂移。

## 速度主指标与双门

- 主指标 = **单条延迟**(batch=1,warmup 后均值);吞吐为辅记录。
- **双门(贯穿全程)**:延迟相对**当前 base** 降 ≥10% **且** 质量相对**最初基线**损失 ≤1%。
- 3a/3c 选型:质量达标候选里**最小延迟者**胜(非"提升 ≥X")。

## 流程(需求文档第三阶段)

唯一真相源 = `runs/<tag>/phase3/state.json`,只经 `scripts/phase3_state.py` 读写。任意时刻先 `resume` 决定续点:
```bash
uv run python -m scripts.phase3_state resume --run-dir runs/<tag>
# → {"action": "init|baseline|framework_select|search|execute|gate_check|parallel|done"}
```

### 步骤 0:Setup(唯一人工关卡)
续用 phase2 `<tag>`;worktree 切 `speedup/<tag>`(共享目录用 worktree,不动共享 HEAD);
钉 `bench_config.json`;在主干原 phase2 镜像上跑 benchmark+evaluate 建 baseline;
`phase3_state init` → `sub_phase=framework-select`。确认后进入自主流程,之后不再逐步问人。

### 模式 3a:单卡基础框架选型(一次性,按 phase1 复现范式)
GitHub 搜本模型领域推理加速框架 → 选 ≤3 排序 → 逐个复现(已有该模型直接接 adapter;
没有则参照相似模型改造)→ benchmark+evaluate → **质量达标里最小延迟者 = base v0**;
无任何达标则退回主干原生推理作 base。进 3b。

### 模式 3b:单卡加速方案优化循环(饱和自动进 3c,可人工触发)
LOOP:搜 github/Arxiv/HF/paperswithcode 选 3 个可复用加速方向(量化/KV-cache/投机解码/
kernel 融合/CUDA graph/torch.compile…,`directions.json` 用 `--tiers phase3` 校验、跨轮去重)→
`dispatch` 按空闲卡并发、训练型(量化校准/草稿训练)排队 → 各在当前 base 上叠加实现 adapter →
benchmark+evaluate 评分 → `gate` 判双门 → 过门 `promote_base`+commit、`dry_streak` 清零、回搜索;
无过门 `bump_dry_streak`;`saturation_check` 到 K 轮 → 进 3c。**绝不停下问人**。

### 模式 3c:多卡并行扩展(一次性,≤3 SOTA 方案)
取最终单卡 base → 选 ≤3 并行方案(TP/PP/EP/SP/replica)→ 逐个多卡实现 adapter →
benchmark(同测单条延迟)+ 质量校验 → **质量达标里最小延迟者 = final** → `sub_phase=done`。交付。

完整细则见 [references/speedup-loop.md](references/speedup-loop.md)。

## 关键纪律
- `dataset/`+`evaluate.py`+`benchmark.py`+`bench_config.json` 全程**不可变**,保证可比。
- **双门 AND**:提速 ≥10%(vs 当前 base)且 质量损失 ≤1%(vs 最初基线,防累积击穿)。
- **keep/discard 用 state.json 指针,不用 git reset**:失败框架/方案目录保留作研究档案。
- 容器/训练输出进 log,只失败时 `tail`;逐实验重试上限 3,crash 不阻塞同轮兄弟。
- 中断可恢复:入口 `resume` 读 state.json;已 scored slot 跳过、已晋升 base 不回退、按 sub_phase 续。
- **容器纪律(继承前阶段)**:每次 `docker run` 带 `--user $UID:$GID --runtime=nvidia`;
  caches 以 `:ro` 挂 `/cache/{modelscope,huggingface,torch}`,env 注入对应 `*_CACHE`/`*_HOME`。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/phase3_state.py {init,resume,gate,dispatch,base-get}` | 状态/双门/派发(确定性核心) |
| `scripts/benchmark.py` | 冻结测速协议:产 speed.json + predictions.jsonl |
| `scripts/directions_schema.py --file <f> --tiers phase3` | 方向 schema 校验(phase3 tier) |
| `scripts/train_launch.py` | 量化校准/草稿训练:数据出处/多卡启动/ckpt 续/预算 |
| `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py` | 复用前阶段 |
