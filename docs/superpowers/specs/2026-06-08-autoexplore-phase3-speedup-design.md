# autoexplore 第一版设计 · 第三阶段(模型推理速度优化)

**日期**: 2026-06-08
**范围**: 需求文档第三阶段步骤 1–4 端到端(3a 单卡框架选型 + 3b 单卡加速方案优化循环 + 3c 多卡并行扩展)
**形态**: 独立的第三个 Claude Code Skill(`autoexplore-phase3`)编排 + Python 工具脚本(混合);在第二阶段产出获胜主干后被调用
**前置**: 第二阶段已在 `runs/<tag>/phase2/state.json` 产出获胜 `backbone`(模型权重 + 其在冻结测试集上的质量分);第一阶段冻结的 `dataset/`、`evaluate.py` 仍在位

## 1. 目标与范围

构建第三阶段「模型推理速度优化」agent,覆盖[需求文档](../../../prd-20260521.md)**第三阶段**全部步骤:取第二阶段获胜模型,进入生产级别的推理速度优化,产出最终多卡推理方案。三个子阶段做成同一状态机里的三个**顺序模式**:

- **3a 单卡基础框架选型**:GitHub 搜本模型领域的推理加速框架,选 ≤3 个 SOTA 按官方披露效果排序,逐个复现推理,**质量达标里单条延迟最小者**作单卡基础框架。
- **3b 单卡加速方案优化循环**:在 base 框架上,搜可信源的跨领域可复用加速方案,选 3 个实现,过**双门**(延迟↓≥10% 且 质量损失≤1%)即叠加为新 base,再进循环;饱和(连续 K 轮无方案过门)自动进 3c。
- **3c 多卡并行扩展**:取最终单卡 base,选 ≤3 SOTA 并行方案,质量达标里**单条延迟最小者**作最终多卡推理方案。

**这是一个通用 skill,不绑定任何具体任务、数据集或模型。** `runs/<tag>/` 里的一切(研究方向、测试集、质量指标、主干模型)都来自前两阶段的运行产物;本设计只依赖**文件契约**(`metrics.json` 的 `{primary_metric, metrics{}}` 形态、`evaluate.py` 的 CLI、`phase2/state.json` 的 backbone),不依赖任何任务语义。

**核心分工**(延续 phase1/2):Skill 负责*判断与编排*(搜框架/方向、按相似模型改造 adapter、调试失败、判断"没主意"时如何破局);Python 脚本负责*确定性重活与易错记账*(base 指针/轮次/实验登记、**双门数学**、测速协议、空闲卡并发派发、饱和判定、中断恢复)。两者通过**文件系统约定**(工作目录 + JSON schema)通信。

### 关键决策(交互确认)

- **覆盖范围**:完整第三阶段 3a + 3b + 3c,单个 skill。
- **速度主指标**:**单条延迟**(batch=1,warmup 后报均值/p50/p99,以**均值**作主排序量);吞吐(throughput_qps)为辅记录。
- **双门(贯穿全程的不变量)**:`speedup ≥ 10%`(vs **当前 base**)**且** `quality_loss ≤ 1%`(vs **最初质量基线**,防多次小损失累积击穿)。
- **3a/3c 选型**:不是"提升 ≥X",而是**质量达标候选里取最小延迟**(PRD"最快者胜")。
- **3b→3c 过渡**:连续 K 轮无方案过门(饱和)**自动进**多卡,亦可人工触发。

### 第一版明确不做(避免 scope 膨胀)

- 多机/集群调度——只在单台 8 卡服务器内按空闲度选卡;3c 在单机多卡上做并行。
- 修改冻结的 `dataset/` 与 `evaluate.py`——它们是质量 ground truth,只读。
- 自动判定"研究方向/质量该不该再优化"——方向与主干质量由前两阶段确定,本阶段只优化推理速度。
- 密钥管理——HF/ModelScope token 由用户预先在环境配好。

## 2. 运行环境假设

- 云 GPU 服务器,已装 Docker + NVIDIA runtime;8 张 GPU,按空闲度动态选卡(复用 `gpu_select.py`)。3c 多卡需 N 卡时等够卡再起。
- **复用前阶段主干模型的 docker 环境**作 baseline 测速基础;框架复现/方案实现需额外依赖时,在主干镜像或框架官方镜像上写派生 Dockerfile 增量构建(`docker_env.py`)。
- 实验有推理型(秒~分钟)与训练型(量化校准、投机解码草稿训练、蒸馏:小时级)两类;训练预算按方向配置(对齐 CLAUDE.md 对本项目的说明)。
- 权重/数据集/校准集下载走 ModelScope > HF,落项目内 `caches/`,不依赖 `~/.cache/`(见 network_sources 经验)。
- **测速可比性**:`bench_config.json` 在 setup 时钉死固定记录子集、warmup/iters、目标卡型;所有框架/方案用**同一把尺**测速。

## 3. 总体架构与目录结构

第三阶段续用同一 `runs/<tag>/`(同方向、同冻结测试集、同质量地基),新增 `phase3/` 子树:

```
runs/<tag>/
├── dataset/  evaluate.py  results.tsv          # phase1 冻结:质量地基(只读)
├── phase2/state.json                            # 取获胜主干来源
└── phase3/
    ├── state.json                               # 唯一真相源(phase3_state.py 独占读写)
    ├── bench_config.json                        # setup 钉死:固定记录子集 / warmup / iters / 目标卡型(只读)
    ├── results.tsv                              # append-only,untracked
    ├── baseline/  speed.json  metrics.json      # 主干在原 phase2 镜像上的 延迟基线 + 质量基线(双门参照)
    ├── frameworks/                              # 3a:每个候选框架一目录
    │   └── <fw>/  adapter.py  Dockerfile?  speed.json  metrics.json  build.log  run.log  progress.json
    ├── single_card/rounds/r001/                 # 3b:单卡优化轮
    │   ├── directions.json  search.log
    │   └── exp_{a,b,c}/  variant.md  adapter.py  Dockerfile?  speed.json  metrics.json  run.log  progress.json
    └── multi_card/                              # 3c:并行方案
        └── scheme_{tp,pp,...}/  variant.md  adapter.py  speed.json  metrics.json  run.log  progress.json
```

### 关键设计点

- **两个冻结地基(全程不可变,保证可比 —— 对齐 autoresearch 最重要原则)**:
  - 质量:沿用 phase1 冻结的 `evaluate.py` + `baseline/metrics.json`(= phase2 主干分);
  - 速度:`benchmark.py`(scripts/ 通用测速协议)+ `bench_config.json`(本 run 钉死的协议参数)+ `baseline/speed.json`。
- **唯一可变部分 = `adapter.py`**:Claude 按每个框架/方案写,暴露统一接口。`benchmark.py` 跑一次**同时**产出 `speed.json`(单条延迟均值/p50/p99 + 吞吐)**和** `predictions.jsonl` → 喂冻结 `evaluate.py` 出 `metrics.json`。**同一次执行同时测速与测质**,杜绝二者配置漂移。
- **base 会演进**:`state.json` 的 `base_framework` 是指针,初值 = 3a 选出的框架;3b 每次过双门叠加新方案后 `version_n++` 改指向。
- **keep/discard 用指针,不用 git reset**(沿用 phase2):失败的框架/方案目录**保留**作研究档案;只有过门才推进 base 指针。
- **日志隔离**:容器/训练输出进 `build.log`/`run.log`/`train.log`,Claude 只在失败时 `tail`,绝不让全量日志淹没上下文。

## 4. 端到端控制流(三模式映射 PRD 第三阶段)

**J** = Claude 判断,**S** = 脚本执行,**H** = 人工确认。状态机 `phase3_state.py` 的 `sub_phase` 在 `framework-select → single-card-loop → multi-card → done` 间推进;任意时刻入口先 `resume` 决定续点。

```
步骤 0 — Setup(唯一人工关卡)                                          [J+H, S]
  续用 phase2 <tag>;worktree 切 speedup/<tag>(共享目录,用 worktree 不动共享 HEAD);
  取 phase2 获胜主干;钉 bench_config.json(固定子集/warmup/iters/卡型);
  在主干「原 phase2 镜像」上跑 benchmark.py + evaluate.py 建 baseline(延迟基线 + 质量基线);
  phase3_state init → sub_phase=framework-select。确认后进入自主流程,之后不再逐步问人。

模式 3a — 单卡基础框架选型(一次性,按 phase1 复现范式)                  [J, S]
  a. [J] GitHub 搜本模型领域的推理加速框架(如 vLLM/TensorRT-LLM/SGLang/LMDeploy…,
         由模型领域决定,通用不写死)→ 按官方披露效果排序 → 选 ≤3 → frameworks 登记。
  b. [J+S] 逐框架复现(框架复现循环,类比 phase1 模型复现):
         build 派生镜像 → 框架已有该模型则直接接 adapter;没有则参照相似模型改造 adapter。
         benchmark.py 一次产 speed.json + predictions → compute_metrics 出 metrics.json。
         复现失败按日志调镜像/adapter 重试(上限 3),仍不行记 crash。
  c. [S] 质量过门(损失 ≤1%)的框架里,单条延迟最小者 = 单卡基础框架(base_framework v0)。
         无任何框架质量达标 → 退回主干原生推理作 base(并记此情形)。进入 3b。

模式 3b — 单卡加速方案优化循环(饱和自动进 3c,可人工触发)              [J, S]
  LOOP:
    a. [J] 在 github/Arxiv/HF/paperswithcode 搜「跨领域可复用」的推理加速方向
           (量化 INT8/FP8/INT4、KV-cache、投机解码、kernel 融合、CUDA graph、torch.compile…)
           → 选 3 → directions.json;跨轮去重(directions_schema 校验)。
    b. [S+J] dispatch 按空闲卡派发 3 个 exp;各在「当前 base 框架」上叠加实现 adapter
             (需校准/训练用 train_launch.py);benchmark.py 出 speed+predictions → metrics。
    c. [S] 各算 speedup% vs 当前 base 延迟、quality_loss% vs 质量基线。
    d. [S+J] 过双门(延迟↓≥10% 且 质量损失≤1%)的最佳者 → 晋升为新 base(叠加,version_n++)
             → commit → dry_streak 清零 → 回 a;无过门者 → 全记 discard/crash,dry_streak++。
    e. [S] saturation check:dry_streak ≥ K → sub_phase=multi-card,进 3c。
       绝不停下问人;「没主意」时(autoresearch 纪律)重读论文/组合近似命中/更激进改动。

模式 3c — 多卡并行扩展(一次性,≤3 SOTA 方案)                          [J, S]
  a. [J] 取最终单卡 base;选 ≤3 SOTA 并行方案(TP/PP/EP/SP/replica,框架而定)。
  b. [J+S] 逐方案在多卡上实现 adapter;benchmark.py 多卡测同一「单条延迟」主指标 + 质量校验。
  c. [S] 质量达标里单条延迟最小者 = 最终多卡推理方案 → final → sub_phase=done。交付。
```

### 数据流约定

- **bench_config.json 是速度可比性契约**:固定记录子集 + warmup/iters + 目标卡型,所有框架/方案同一把尺。
- **directions.json 沿用 phase2 schema**,扩展:`tier ∈ {framework, quantization, kernel, decoding, compile, parallel}`、`expected_speedup`、`quality_risk`。
- **双门是晋升契约**:`speedup ≥ 0.10`(vs **当前 base**)**且** `quality_loss ≤ 0.01`(vs **最初基线**)才接受;3a/3c 选型则在「质量达标候选」里取最小延迟。
- **speed/predictions/metrics 复用前阶段格式**:benchmark.py 产 `speed.json` + `predictions.jsonl`(phase1 schema),compute_metrics 调冻结 `evaluate.py` → 归一 `{primary_metric, metrics{}}`。
- **失败也写 results.tsv**:未过门 = `discard`,崩 = `crash`,不阻塞同轮兄弟。

## 5. worker 组件细则

### 5.1 `benchmark.py` — 冻结测速协议(scripts/ 通用工具)

把"单条延迟怎么测才可比"钉成一段标准协议,**绝不混入任何框架/任务语义**(那些在 adapter 里):

- 入参:`--adapter <path> --bench-config bench_config.json --dataset <dir> --out speed.json --predictions predictions.jsonl`。
- 协议:`load_model()` 一次 → **warmup N 条**(丢弃)→ 对 `bench_config` 钉死的固定记录子集,**batch=1 逐条** `infer_one(record)`,每条 `torch.cuda.synchronize()` 后用 `perf_counter` 计时,重复 iters 轮取每条多次测量。
- 产出 `speed.json`:`{latency_mean_ms, p50_ms, p99_ms, throughput_qps, n_records, warmup, iters, framework, gpu_name, gpu_count}`;**同时**把每条输出写 `predictions.jsonl`(复用 phase1 schema)→ 一次执行喂 compute_metrics 测质。
- 多卡(3c):同协议测**单条延迟**主指标(TP/PP 降单条延迟;replica 主要抬吞吐——吞吐记入辅指标,主排序仍是单条延迟)。

### 5.2 adapter 契约(唯一可变部分,Claude 按框架/方案写)

```python
# adapter.py —— 框架/方案特定,benchmark.py 在容器内 import
def load_model(config: dict): ...                 # 加载一次,返回 handle(权重/引擎/并行组)
def infer_one(handle, record: dict) -> dict: ...  # 单条推理,返回与 phase1 predictions 同 schema 的一条
```

框架已有该模型 → adapter 薄薄一层包官方 API;没有 → Claude 参照相似模型改造。量化/投机解码草稿等需校准/训练的,先用 `train_launch.py` 产物(量化权重/草稿模型)落 `exp_*/`,adapter 再加载之。

### 5.3 框架/方案发现(Claude 驱动 + 薄 schema 校验)

Claude 用 WebSearch/WebFetch 检索:3a 在 GitHub 搜本模型领域推理加速框架;3b 在 github/Arxiv/HF/paperswithcode 搜跨领域可复用加速方案。通用排序准则:报告加速比、与本模型适配成本(已有该模型代码 vs 需改造)、质量风险、实现 tier 成本。产出 `directions.json`(schema 见 §4),`directions_schema.py` 仅校验/规范结构,**搜索本身是 Claude 的判断**。

### 5.4 实验执行(复用前阶段脚本)

每框架/方案一个工作目录,`variant.md` 写清改动。docker 按需 build 派生镜像(`docker_env.py`)。按 tier 实现 `adapter.py`;需校准/训练用 `train_launch.py` 产物。`benchmark.py → speed.json + predictions` → `compute_metrics → metrics.json`;逐实验 `progress.py` 记状态,重试上限 3。并发由 `dispatch plan` 按空闲卡分配,训练型排队。

### 5.5 `train_launch.py` — 通用训练壳(复用 phase2)

第三阶段用于:量化校准(下论文/官方校准集,无则类似公开集)、投机解码草稿模型训练、蒸馏。标准化数据集获取(出处记 `exp_*/dataset_provenance.json`,ModelScope>HF)、多卡启动(输出→`train.log`)、checkpoint 续训、预算+监控+恢复。

## 6. `phase3_state.py` — 确定性受测核心

独占读写 `state.json`,把最易错的**双门数学**、基线参照、并发派发、饱和判定、中断恢复钉成纯逻辑 + CLI(仿 phase2_state):

| 子命令 | 职责 |
|--------|------|
| `init` | 取 phase2 主干 + 建 baseline 占位 + 钉 sub_phase=framework-select |
| `baseline set` | 写 `baseline.quality` / `baseline.latency_ms` / `baseline.throughput_qps`(双门永久参照) |
| `framework add` / `framework record` | 3a 候选登记 / 记测速+质量结果 |
| `base set` / `base get` | 选 / 读单卡 base 指针(3b 演进,`version_n++`、记 history) |
| `round open` / `round record` | 3b 轮(rNNN)与 slot 登记 |
| `gate check` | **纯函数双门**:`speedup = (base_lat − cand_lat)/base_lat ≥ 0.10` 且 `quality_loss = (baseline_q − cand_q)/baseline_q ≤ 0.01` → accept;边界单测 |
| `dispatch plan` | 输入=就绪实验+卡需求+GPU 快照 → 输出分配;非训练优先、训练型卡紧排队(接收快照所以可测) |
| `saturation check` | `dry_streak ≥ K` → 推 sub_phase=multi-card |
| `parallel add` / `parallel record` | 3c 方案登记;质量达标里最小延迟 → final |
| `directions add` / `directions seen` | 跨轮已试方向登记与去重 |
| `resume` | 读 state 告诉 Claude 从 哪个 sub_phase + 哪步(framework/search/score/promote/parallel)续 |

### state.json schema(支持中断恢复)

```json
{
  "tag": "<run-tag>",
  "baseline": {"source": "phase2-backbone:<name>", "quality": 0.0, "latency_ms": 0.0, "throughput_qps": 0.0},
  "sub_phase": "framework-select",            // framework-select|single-card-loop|multi-card|done
  "frameworks": [{"name":"vllm","exp_dir":"frameworks/vllm","status":"ready",
                  "latency_ms":0.0,"quality":0.0,"passes_quality":true}],
  "base_framework": {"name":"vllm","exp_dir":"frameworks/vllm","latency_ms":0.0,
                     "quality":0.0,"version_n":0},
  "base_history": [{"version_n":0,"name":"vllm","latency_ms":0.0,"promoted_at":"..."}],
  "round_counter": 0,
  "rounds": [{"id":"r001","status":"scored",
              "slots":[{"slot":"a","exp_dir":"single_card/rounds/r001/exp_a","tier":"quantization",
                        "latency_ms":0.0,"quality":0.0,"speedup_pct":0.0,
                        "quality_loss_pct":0.0,"status":"discard"}]}],
  "dry_streak": 0, "saturation_k": 3,
  "directions_tried": ["<规范化指纹>"],
  "parallel_schemes": [{"name":"tp2","exp_dir":"multi_card/scheme_tp","status":"ready",
                        "latency_ms":0.0,"quality":0.0}],
  "final": null,                               // 完成后:{"scheme":..., "latency_ms":..., "quality":..., "gpu_count":...}
  "updated_at": "2026-06-08T..."
}
```

### 门的明确决定

- **双门 `AND`**:速度提升 ≥10%(vs **当前 base**)且 质量损失 ≤1%(vs **最初基线**,防累积击穿)。
- **3a/3c 选型**:不是"提升 ≥X",而是**质量达标候选里取最小延迟**(PRD"最快者胜")。
- **边界**(`speedup==10%`、`quality_loss==1%`、base 延迟为 0、质量基线为 0)在 `gate check` 纯函数单测里钉死。

## 7. 错误处理与边界

| 失败类型 | 处理层 | 策略 |
|---------|-------|------|
| 框架镜像/依赖构建失败 | 框架复现(3a) | 计入 retry_count,Claude 读 build.log 改派生 Dockerfile;上限 3 仍败 → 该框架 crash,选下一个 |
| 框架无该模型代码 | adapter | 参照相似模型改造 adapter;无相似可参照 → 记 crash 标因 |
| 框架改了数值致质量超损 | `gate`/选型 | `passes_quality=false`,排除出 base 候选(快但不达标无意义) |
| 量化校准/草稿训练失败 | `train_launch.py` | 区分可重试(网络/OOM 降配)与不可重试(无公开校准集);后者该方向 crash 标因 |
| 测速噪声/抖动 | `benchmark.py` | warmup + 多 iters 取稳定统计;同卡型复测;`bench_config` 固定子集保证可比 |
| GPU 全忙/卡不足 | `dispatch plan` | 训练型排队、非训练优先;3c 多卡需 N 卡时等够卡再起 |
| 推理崩溃 | 实验执行 | 有界重试 3 |
| `evaluate.py` 报错 | `compute_metrics` | 该实验测质失败 `status=crash`,不阻塞同轮兄弟 |
| 单 slot 崩溃 | 轮编排 | 该 slot=crash,另两个继续,存活者算分判门 |
| 搜索无可用结果 | 方向发现 | 拓宽检索 / "没主意"纪律(重读论文、组合近似命中、更激进改动),绝不卡死 |
| 整个 run 中断 | `resume` | 读 state.json + 逐实验 progress.json + ckpts:已 scored slot 跳过、已晋升 base 不回退、按 sub_phase 续 |
| 3b 久不过门 | `saturation check` | dry_streak≥K 自动进 3c,不无限空转 |

**有界重试上限 3**(对齐 phase1/2 与 CLAUDE.md);crash 不阻塞链路;晋升只在过双门时发生。

## 8. 测试策略

Python 标准库 + pytest,镜像 phase1/2 布局(unit + contract + gpu-marked smoke)。脚本可独立测试;Claude 判断部分靠 speed/metrics/variant/results 留痕兜底。

### 单元测试(不碰真实 Docker/GPU)

| 脚本 | 测什么 | 怎么测 |
|------|--------|--------|
| `phase3_state.py` | **双门边界**(speedup==10%、quality_loss==1%、base/基线为 0)、base 初始化与晋升往返、轮/slot 登记、saturation(dry_streak≥K)、dispatch 训练排队、directions 去重、3c 选型取最小延迟、resume 决策、state.json 原子写 | tmp 假 run + 假 phase2 state;门函数喂边界值;dispatch 喂 mock GPU 快照断言分配 |
| `benchmark.py` | warmup/iters 计时逻辑、固定子集选取、百分位计算、speed.json+predictions 双产出、缺 adapter 接口优雅报错 | 玩具 adapter(纯 Python sleep stub)+ 微型 dataset,断言统计字段与 predictions schema |
| `directions_schema.py` | 新 tier 枚举 + 字段校验 | 合法/非法样例(复用并扩展 phase2 的) |

### 契约测试(脚本间文件约定)

端到端 *dry-run*:玩具 dataset + 玩具 evaluate.py + stub docker/adapter,模拟 3a→3b→3c 一整条文件流转(baseline → frameworks 选 base → 一轮 single_card 过双门晋升 → multi_card 选 final → state.json/results.tsv),断言每步产物 schema 与字段对得上。锁住跨脚本接口——最有价值的测试。

### 冒烟测试(需真实环境,默认 skip,`@pytest.mark.gpu`)

微型主干上:一个框架走 build→adapter→benchmark→质量校验;一个 infer-only 方案走 dispatch→benchmark→双门;3c 一个最小 TP 方案走多卡 benchmark。

### 不测

- Claude 判断质量(框架/方向搜索与排序、adapter 改造代码)——靠 speed.json/metrics.json/variant.md/results.tsv 留痕人工复核。
- 真实加速幅度——agent 运行结果,非单元测试范畴。

**TDD 节奏**:每个脚本先写测试(红)→ 实现(绿)→ 重构,小步提交。

## 9. 交付物与 git 工作流

### 交付物

- `skills/autoexplore-phase3/SKILL.md`(独立第三个 skill,与 phase1/2 并列;phase2 产出获胜主干后被调用)
- `skills/autoexplore-phase3/references/speedup-loop.md`(三模式细则,从 SKILL.md 引用)
- 新脚本:`phase3_state.py` / `benchmark.py`(冻结测速协议)
- 复用:`gpu_select.py` / `docker_env.py` / `run_inference.py` / `compute_metrics.py` / `progress.py` / `directions_schema.py`(扩 tier) / `train_launch.py`
- `tests/` 镜像 phase1/2 结构(test_phase3_state / test_benchmark / 扩 directions_schema / contract dry-run / gpu smoke)

### git 工作流

- **构建 skill 本身**:走 `GIT_CONVENTIONS.md`,从**当前工作分支** `autoexplore-infer` 切 `feature/phase3-speedup`(依经验:从当前分支切,不从落后的 develop;phase3 复用 phase1/2 脚本,当前分支已具备);完成后按约定 squash 合回。**共享目录用 worktree**,不动共享 HEAD。
- **skill 运行时的优化循环**:在专用分支 `speedup/<tag>` 上跑,每个过门方案直接 commit 留痕;keep/discard 由 state.json 指针管理,不用 git reset。
