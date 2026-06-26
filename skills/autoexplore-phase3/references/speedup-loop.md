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
1. 取最终单卡 base;选 ≤3 SOTA 并行方案(框架而定)。**主指标是单条延迟**,只有 **TP/SP**
   能降单条延迟;**replica/数据并行只增吞吐,不降单条延迟**——别拿它当延迟方案。
2. 逐方案(`multi_card/scheme_<name>/`):多卡实现 adapter(`dispatch` 要够 N 卡),
   benchmark(同测单条延迟主指标)+ compute_metrics;`parallel_add`/`parallel_record`。
   - **框架无原生多卡推理时自实现 TP/SP**:并行逻辑写进 adapter(单进程 benchmark 测不了多进程),
     用 `scripts/benchmark_dist.py` 桥接——torchrun 起 N 进程、复用冻结的 `benchmark.measure`、
     各 rank 跑同一固定子集**锁步**(collective 自然同步;固定种子→各 rank 输出一致)、**仅 rank0 写盘**。
     这样协议与单卡完全一致、可比。
   - **collective(all-reduce 等)用 fp32 累加**:低精度累加误差随卡数增长,world≥4 时常击穿 1% 质量门。
   - **显式 per-rank 设备**:有的加载器对 `device="cuda"` 不认 `set_device`,会把各 rank 都堆到 0 卡 → 传 `cuda:{local_rank}`。
   - **docker 多卡**:`--gpus device=a,b` 可能报 "cannot set both Count and DeviceIDs" → 改用 `NVIDIA_VISIBLE_DEVICES`。
   - 并行延迟收益**受通信限制、次线性递减**;遍历几个卡数,取"质量达标里最小延迟",别默认越多越快。
3. `select_final`:质量达标里最小延迟者 = final(`gpu_count` 随之),无则回落单卡 base。
   `sub_phase=done`。交付最终生产级多卡推理方案。

## 错误处理速查
- 镜像/依赖构建失败:retry≤3,读 `build.log` 改派生 Dockerfile,仍败记 crash。
- 框架无该模型:参照相似模型改 adapter;无可参照记 crash 标因。
- 框架数值改动致质量超损:`passes_quality=false`,排除出 base 候选。
- 测速抖动:加大 warmup/iters,固定子集复测。
- 卡不足:训练型/多卡排队,非训练优先。
- 中断:`resume` 按 sub_phase 续,已 scored slot 跳过、已晋升 base 不回退。

## 通用工程坑速查(跨任务复用)
与具体任务/模型无关、最易踩且最该提前规避的工程坑:

### 测速可比性(多租户/共享节点尤甚)
- **比"固定子集均值",不信单点快照**:逐迭代耗时随 GPU 降频/同租户竞争抖动很大,但固定子集
  的**均值跨卡稳定**(抖动会被整轮平均掉)。同一固定子集的均值才是可比量,gate 也判均值。
- **p50→p99 大跨度未必是抖动**:常是真实的逐样本计算量差异(输入尺寸/序列长度不同)。
  别误判为不稳定去乱调 warmup;均值仍可比。
- **子集大小权衡**:严格 ≤1% 质量门配小子集,在阈值附近**噪声大、判定脆**;
  钉子集时在"测速够快"与"质量门够准"之间取平衡(越激进的有损杠杆越需要更大子集)。
- 质量与算力竞争**无关**:质量可在任意卡先筛,过门候选再到干净卡补测延迟。

### compile / kernel 杠杆(compile tier)
- **容器内编译后端会调 getpass/getuser**:以非 passwd 内 UID 跑容器会崩 → 注入
  `USER`/`LOGNAME`/`HOME` 与可写 cache 目录(`*_CACHE_DIR`/`XDG_CACHE_HOME`)。
- **动态输入形状会在计时窗内重编译**,污染延迟 → 在 `load_model`(不计时)里**预热所有不同形状**,
  让计时窗零重编译;预热须用与计时相同的关键参数(如 batch/guidance 模式),否则编译的是另一条图。
- **挂持久化 compile cache**(host 目录 → 容器),避免每次从头编译;激进 autotune 模式编译很慢,要留预算。

### 缓存挂载与加载器
- caches 优先 `:ro`,但**下载器/加载器常需写锁**(.lock)即便权重已缓存 → 该 cache 放开为 RW。
- 本地权重复用:让加载器从本地目录解析(cwd/显式路径),避免每次走网络。

(多卡 TP/SP 的坑见上文"模式 3c";要点:adapter 内自实现 + `benchmark_dist.py` 桥接、collective 走 fp32、
显式 `cuda:{local_rank}`、`NVIDIA_VISIBLE_DEVICES`、收益次线性。)
