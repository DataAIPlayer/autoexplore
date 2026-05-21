---
name: autoexplore-phase1
description: Use when reproducing open-source models for a research direction — drives the full phase-1 pipeline (clarify direction, build test set, design metrics, search/rank models, reproduce in Docker, infer, score, pick best).
---

# autoexplore 第一阶段:开源模型自动复现

输入:一段研究方向描述。产出:在该方向上复现并评测 ≤3 个开源模型,选出指标最高者。

环境:云 GPU 服务器,Docker + NVIDIA runtime,8 卡,按空闲度自动选卡。

## 流程(需求文档第一阶段步骤 1–10)

### 步骤 1–3:与用户确认(唯一的人工关卡)

1. **澄清方向**:与用户对话明确研究方向。商定一个 run-tag(如 `vqa-mar5`),创建 `runs/<tag>/`,写 `plan.md` 草稿。
2. **建立测试集**:按方向提议数据来源与规模,**人工确认后**落到 `runs/<tag>/dataset/`。此后不可变。
3. **设计指标**:写 `runs/<tag>/evaluate.py`,接口为
   `python evaluate.py --predictions <jsonl> --dataset <dir> --out <metrics.json>`,
   输出 `{"primary_metric": <float>, "metrics": {...}}`。**人工确认后冻结。**
   在 `plan.md` 里约定 predictions.jsonl 的每行 JSON 字段(供推理脚本与 evaluate.py 共同遵守)。

把步骤 1–3 的结论都写进 `plan.md`,得到用户确认后再进入步骤 4。之后尽量自主,不再逐步问人。

### 步骤 4–5:搜索与排序模型

4. **搜索**:在 Hugging Face、paperswithcode、arxiv 检索该方向可用开源模型。
5. **排序**:按官方披露效果自高到低排序,选 ≤3 个,写 `runs/<tag>/candidates.json`:
   ```json
   [{"name": "...", "repo": "...", "reported_score": 0.0, "priority": 1}]
   ```

### 步骤 6–9:逐个复现 + 推理 + 评测

按 priority 升序,对每个候选执行**复现循环**。完整细则见
[references/reproduction-loop.md](references/reproduction-loop.md):准备环境(选卡、构建/复用镜像)→ 下载 → 验证推理 → 有界重试(上限 3 次)→ 推理产出 predictions → compute_metrics 算分 → 写 results.tsv。

### 步骤 10:选最佳模型

读 `runs/<tag>/results.tsv`,在 `status=ready` 的模型里选 `primary_metric` 最高者,向用户报告。该模型是第二阶段(效果迭代)的候选。

## 关键纪律

- `dataset/` 与 `evaluate.py` 人工确认后**不可变**,保证模型间可比。
- 容器输出进 `run.log`,只在失败时 tail,不污染上下文。
- 每模型重试上限 3 次,失败标 `crash` 但不阻塞其他模型。
- 复现串行,一次一个(第一版不并行、不训练)。
- 中断可恢复:复现循环开始读 `progress.json`,跳过已 ready/crash 的模型。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/gpu_select.py --count N --min-free-mib M` | 选卡 → 打印 CUDA_VISIBLE_DEVICES |
| `scripts/docker_env.py {check,build,run}` | Docker 检查 / 镜像构建复用 / 容器执行 |
| `scripts/run_inference.py` | 容器内推理 → predictions.jsonl |
| `scripts/compute_metrics.py` | 调 evaluate.py → metrics.json |
| `scripts/progress.py {set,done,result}` | 进度持久化 / 终态查询 / 结果汇总 |
