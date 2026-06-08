# 推理速度优化循环细则(第三阶段)

承接 SKILL.md。所有路径相对 repo 根;命令均从 repo 根 `uv run`。
状态唯一真相源 = `runs/<tag>/phase3/state.json`,只经 `scripts/phase3_state.py` 读写。
进入任意阶段前先 `resume`:
```bash
uv run python -m scripts.phase3_state resume --run-dir runs/<tag>
# → {"action": "init|baseline|framework_select|search|execute|gate_check|parallel|done"}
```

## 步骤 0:Setup(唯一人工关卡)
1. 续用 phase2 `<tag>`;`git worktree add .claude/worktrees/speedup-<tag> -b speedup/<tag> HEAD`(共享目录纪律)。
2. 钉 `runs/<tag>/phase3/bench_config.json`,例:
   ```json
   {"subset_ids": ["<固定挑选的代表性样本 id>"], "warmup": 5, "iters": 20,
    "gpu_name": "<目标卡型>", "gpu_count": 1, "framework": "baseline",
    "model_config": {"<主干推理所需参数>": "..."}}
   ```
   子集要固定且有代表性;warmup/iters 取够稳(p99 不抖)。**钉死后不再改**。
3. 建 baseline(主干原生推理,原 phase2 镜像):写 `baseline/adapter.py` → 测速+评分:
   ```bash
   uv run python -m scripts.benchmark --adapter runs/<tag>/phase3/baseline/adapter.py \
     --bench-config runs/<tag>/phase3/bench_config.json --dataset runs/<tag>/dataset \
     --out runs/<tag>/phase3/baseline/speed.json \
     --predictions runs/<tag>/phase3/baseline/predictions.jsonl
   uv run scripts/compute_metrics.py --evaluate-py runs/<tag>/evaluate.py \
     --predictions runs/<tag>/phase3/baseline/predictions.jsonl \
     --dataset runs/<tag>/dataset --out runs/<tag>/phase3/baseline/metrics.json
   ```
4. `uv run python -m scripts.phase3_state init --run-dir runs/<tag> --tag <tag>`,再用
   `baseline/speed.json` 的 `latency_mean_ms`/`throughput_qps` 与 `baseline/metrics.json` 的
   `primary_metric` 经 `set_baseline` 写入。确认后进入自主流程。

## 模式 3a:单卡基础框架选型
1. GitHub 搜本模型领域推理加速框架(WebSearch/WebFetch),按官方披露加速比排序,选 ≤3。
2. 逐框架(`frameworks/<fw>/`):
   - `docker_env.py build` 出框架镜像(官方镜像或 `FROM` 派生);
   - 框架已有该模型 → adapter 薄包官方 API;没有 → 参照相似模型改造 adapter;
   - benchmark + compute_metrics;`framework_record` 记 latency/quality/status;
   - 失败读 `build.log`/`run.log` 改镜像或 adapter,重试上限 3,仍败记 crash 换下一个。
3. 全部记完 → `select_base`:质量达标里最小延迟者 = base v0;无则退回主干原生。进 3b。

## 模式 3b:单卡加速方案优化循环(LOOP,饱和自动进 3c)
### a. 搜方向
WebSearch/WebFetch 在 github/Arxiv/HF/paperswithcode 搜**跨领域可复用**加速方向,选 3,
写 `single_card/rounds/rNNN/directions.json`(必填字段同 phase2,`tier` ∈ phase3 集;
可加 `expected_speedup`/`quality_risk` 作排序参考——这两个是参考性字段,schema 不强制校验),
校验并去重:
```bash
uv run scripts/directions_schema.py --file <directions.json> --tiers phase3
```
跨轮去重:用 `directions_seen` 跳过已试方向(标题+URL 指纹)。
### b. 派发与实现
```bash
uv run python -m scripts.phase3_state dispatch \
  --experiments '[{"slot":"a","needs_gpus":1,"is_training":false}, ...]' --free-gpus 0,1,2
```
各 `exp_*/` 在**当前 base** 上叠加实现 `adapter.py`(需校准/草稿训练用 `train_launch.py` 产物);
benchmark + compute_metrics;`open_round`/`record_slot` 记分(自动算 speedup% vs base、loss% vs 基线)。
### c. 双门与晋升
对最佳 slot:
```bash
uv run python -m scripts.phase3_state gate --base-lat <当前base延迟> --cand-lat <候选延迟> \
  --baseline-q <最初基线质量> --cand-q <候选质量>     # → {"accept": bool}
```
- accept → `promote_base`(version_n++、`dry_streak` 清零)+ `git commit` 留痕 → 回 a;
- 无过门 → `append_phase3_result` 记 discard/crash + `bump_dry_streak`。

判定与记账完成后 `close_round` 关闭该轮(状态→done),`resume` 即指向下一轮 search。
### d. 饱和
```bash
# dry_streak ≥ saturation_k(默认 3)→ 自动进多卡;也可人工 force
```
`saturation_check`(force 可人工提前)推进 `sub_phase=multi-card`。**绝不停下问人**;
"没主意"时重读论文、组合近似命中、试更激进改动(对齐 autoresearch 纪律)。

## 模式 3c:多卡并行扩展
1. 取最终单卡 base;选 ≤3 SOTA 并行方案(TP/PP/EP/SP/replica,框架而定)。
2. 逐方案(`multi_card/scheme_<name>/`):多卡实现 adapter(`dispatch` 要够 N 卡),
   benchmark(同测单条延迟主指标)+ compute_metrics;`parallel_add`/`parallel_record`。
3. `select_final`:质量达标里最小延迟者 = final(`gpu_count` 随之),无则回落单卡 base。
   `sub_phase=done`。交付最终生产级多卡推理方案。

## 错误处理速查
- 镜像/依赖构建失败:retry≤3,读 `build.log` 改派生 Dockerfile,仍败记 crash。
- 框架无该模型:参照相似模型改 adapter;无可参照记 crash 标因。
- 框架数值改动致质量超损:`passes_quality=false`,排除出 base 候选。
- 测速抖动:加大 warmup/iters,固定子集复测。
- 卡不足:训练型/多卡排队,非训练优先。
- 中断:`resume` 按 sub_phase 续,已 scored slot 跳过、已晋升 base 不回退。
