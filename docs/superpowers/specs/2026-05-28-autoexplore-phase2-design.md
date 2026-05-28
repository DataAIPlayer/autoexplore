# autoexplore 第一版设计 · 第二阶段(模型效果迭代优化)

**日期**: 2026-05-28
**范围**: 需求文档第二阶段步骤 1–3 端到端(不终止的效果优化循环)
**形态**: 独立的第二个 Claude Code Skill(`autoexplore-phase2`)编排 + Python 工具脚本(混合);在第一阶段产出主干后被调用
**前置**: 第一阶段已在 `runs/<tag>/` 产出冻结的 `dataset/`、`evaluate.py`、`results.tsv` 与至少一个 `status=ready` 的复现模型

## 1. 目标与范围

构建第二阶段「模型效果迭代优化」agent,覆盖[需求文档](../../../需求文档-20260521.md)**第二阶段**全部步骤:以第一阶段选中的模型为**模型主干(backbone)**,分析其在冻结测试集上的短板;先评估"仅调推理管道"的便宜改善空间;再进入**不终止的搜索优化循环**——在可信源(Arxiv / Hugging Face / paperswithcode)搜针对短板的方向、选 3 个实现(需要就训练)、在测试集评分,**有明显提升就把它作为新主干并重新进入循环**,直到人工触发关闭。

**这是一个通用 skill,不绑定任何具体任务、数据集或模型。** `runs/<tag>/` 里的一切(研究方向、测试集、指标、主干模型)都来自第一阶段的运行产物;本设计只依赖第一阶段冻结的**文件契约**(`metrics.json` 的 `{primary_metric, metrics{}}` 形态、`evaluate.py` 的 CLI、`results.tsv` 列),不依赖任何任务语义。

**核心分工**(延续第一阶段):Skill 负责*判断与编排*(读诊断定短板、搜索与排序方向、按 tier 写实现代码、调试失败、判断"没主意"时如何破局);Python 脚本负责*确定性重活与易错记账*(主干指针/轮次/实验登记、晋升门数学、空闲卡并发派发、中断恢复、短板分解、训练壳)。两者通过**文件系统约定**(工作目录 + JSON schema)通信。

### 第一版明确不做(避免 scope 膨胀)
- 多机/集群调度——只在单台 8 卡服务器内按空闲度选卡。
- 修改第一阶段冻结的 `dataset/` 与 `evaluate.py`——它们是 ground truth,只读。
- 自动判定"研究方向是否该换"——方向由第一阶段确定,本阶段只优化既定方向上的主干。
- 副指标守卫晋升——本版晋升只认主指标(见 §6 晋升门)。
- 密钥管理——HF/ModelScope token 由用户预先在环境配好。

## 2. 运行环境假设

- 云 GPU 服务器,已装 Docker + NVIDIA runtime;8 张 GPU,按空闲度动态选卡(复用第一阶段 `gpu_select.py`)。
- **复用第一阶段主干模型的 docker 环境**作为实验基础镜像;某方向需额外依赖时,在主干镜像上写 `FROM <主干镜像>` 的派生 Dockerfile 增量构建。
- 实验有推理型(秒~分钟)与训练型(小时~天)两类;训练预算相对 autoresearch 的固定 5 分钟**放宽**为多卡 / 可变 / 按方向配置(对齐 CLAUDE.md 对本项目的说明)。
- 权重/数据集下载走 ModelScope > HF,落项目内 `caches/`,不依赖 `~/.cache/`(见 network_sources 经验)。

## 3. 总体架构与目录结构

第二阶段续用同一 `runs/<tag>/`(同方向、同冻结测试集),新增 `phase2/` 子树:

```
runs/<tag>/
├── dataset/  evaluate.py  results.tsv     # 第一阶段冻结产物(只读地基)
├── models/<p1-name>/                       # 第一阶段复现的模型(初始主干来源)
└── phase2/
    ├── state.json                          # 受测核心唯一真相源:主干指针 / 轮次 / 实验登记 / 已试方向
    ├── results.tsv                          # 第二阶段实验日志(append-only,untracked)
    ├── diagnostics/
    │   └── diag_<backbone_id>.{json,md}     # 每个主干版本的短板报告(json 给脚本,md 给 agent)
    └── rounds/
        └── r001/
            ├── directions.json              # 本轮 3 方向 + 搜索出处(arxiv/hf/pwc 链接)
            ├── search.log
            └── exp_{a,b,c}/                 # 一方向一工作目录,彼此隔离
                ├── variant.md               # 这个方向改了什么(人/agent 可读)
                ├── Dockerfile?              # 仅当需在主干镜像上加依赖
                ├── <impl 代码>              # 按 tier:config 参数 / postprocess / pipeline / train
                ├── ckpts/?                  # 训练型的 checkpoint(可续训)
                ├── train.log? run.log       # 输出隔离,只失败时 tail
                ├── predictions.jsonl        # 推理产出(复用第一阶段 schema)
                ├── metrics.json             # compute_metrics 产出
                └── progress.json            # 逐实验复现状态(复用第一阶段 progress schema)
```

### 关键设计点
- **主干会演进**:`state.json` 的 `backbone` 是指针,初值=第一阶段 `results.tsv` 中 `status=ready` 且 `primary_metric` 最高者;每次显著晋升后 `version_n++` 并改指向。主干来源可以是第一阶段的 `models/<name>/`,也可以是第二阶段某个 `exp_*/`。
- **实验隔离**:每方向在独立 `exp_*/`,互不干扰,可独立重试/恢复/留痕。
- **keep/discard 用指针,不用 git reset**:实验各自隔离(不像 autoresearch 改同一个文件),"discard" = 不移动主干指针即可;实验目录**保留**作为失败档案(类比 results.tsv 的 crash 行)。只有显著晋升才推进指针。理由:loop forever 下,被 reset 抹掉的隔离实验目录正是宝贵的"试过什么/为何没用"研究档案;用指针管 keep/discard 更安全、可恢复、不丢记录。
- **冻结地基只读**:`dataset/` 与 `evaluate.py` 全程不可变,保证所有主干版本与实验可比(对齐 autoresearch 最重要原则)。
- **日志隔离**:容器/训练输出进 `run.log`/`train.log`,Claude 只在失败时 `tail`,绝不让全量日志淹没上下文。

## 4. 端到端控制流(步骤映射 + 优化螺旋)

**J** = Claude 判断,**S** = 脚本执行,**H** = 人工确认。

```
步骤 0 — Setup(唯一人工关卡)                                            [J+H, S]
  续用第一阶段 <tag>;切专用分支 optimize/<tag>;
  phase2_state init → backbone = results.tsv 中 status=ready 且 primary_metric 最高者;
  校验 dataset/ 与 evaluate.py 在位且冻结。确认后进入自主循环,之后不再逐步问人。

步骤 1 — 短板诊断(每次主干变化都重跑)                                    [S→J]
  diagnose.py 把冻结 evaluate.py 当黑盒,在子集视图上反复调用,产出
  diag_<backbone_id>.json(全集 / 逐样本 / 最差 K / 按 metadata 自动探测的分类字段分组 / 副指标小结)。
  Claude 读报告,把短板小结写进 diag_<backbone_id>.md。

步骤 2 — 推理管道调优闸门(便宜档,先于训练)                               [J, S]
  Claude 判断:仅调主干推理管道(参数/后处理,无新方法、不训练)是否可能改善短板?
    有空间 → 跑有界几个 tier=infer-tune 实验(走步骤 3 同一套执行+评分+晋升);
            任一清过晋升门 → 晋升主干 → 回【步骤 1】重诊断;
    无空间或都没过 → state.json 记 inference_tuning=explored,进【步骤 3】。
  纪律:便宜档永远先穷尽,再动昂贵搜索。

步骤 3 — 不终止搜索优化循环(复用主干 docker 环境)                         [J, S]
  LOOP(直到人工终止):
    a. [J] 在 Arxiv/HF/paperswithcode 搜针对当前短板的方向 → 排序 → 选 3 → directions.json;
           跨轮查 state.json 的 directions_tried 去重。
    b. [S+J] phase2_state dispatch plan 按空闲卡派发 3 个实验(非训练优先、训练型卡紧时排队);
             每个 exp_*/ 按 tier 实现(config/post-process/pipeline/train),需要就 train_launch.py;
             run_inference → predictions;compute_metrics → metrics.json;逐实验重试上限 3。
    c. [S] 三者算分,phase2_state 算各自 delta% vs 当前主干。
    d. [S+J] 最佳者过晋升门(gate check)→ backbone promote(version_n++)、commit → 回【步骤 1】;
             否则三者全记 discard/crash,保持原主干,回 a 选新方向继续。
    e. 绝不停下问人。"没主意"时(autoresearch 纪律):重读论文、组合差一点的近似命中、试更激进改动——不停。
```

### 数据流约定
- **诊断报告是短板契约**:`diag_*.md` 是 Claude 选搜索方向的输入;`diag_*.json` 让脚本/复盘可机读。
- **directions.json 是搜索产物契约**:`[{slot, title, source_urls[], idea, tier, needs_training, dataset_plan, weakness_addressed, est_cost}]`;Claude 写,薄 schema 脚本校验。
- **predictions / metrics 复用第一阶段格式**:`run_inference.py` 产 `predictions.jsonl`,`compute_metrics.py` 调冻结 `evaluate.py` → 归一 `{primary_metric, metrics{}}`。
- **晋升螺旋**:主干一晋升即回步骤 1 重诊断,形成"诊断 → 便宜调优 → 搜索 → 晋升 → 再诊断"螺旋。
- **失败也写进 results**:实验未达门 = `discard`,崩 = `crash`,都记一行,不阻塞同轮兄弟。

## 5. worker 组件细则

### 5.1 `diagnose.py` — 短板诊断(增量、只读、不改冻结 evaluate.py)
把 `evaluate.py` 当**黑盒**(只靠 `--predictions --dataset --out` 契约),对当前主干的 predictions 在**子集视图**上反复调用,**绝不重实现指标**(否则既偏离冻结指标、又绑死任务语义):
- **全集**(对账 metrics.json)→ **逐样本**(单样本视图 → per-sample primary → 最差 K)→ **分组**(按 metadata.json 样本里**自动探测到的分类字段**,有就分、无则优雅降级)。
- 视图 = 临时目录,放子集化的 metadata.json + 指向原样本目录的 symlink;原 `dataset/` 不动。
- 产出 `diag_<id>.json`:`{full, per_sample:[{id,primary}], worst_k, groups:{field:{value:primary}}, secondary_summary}` + 给 agent 读的 `.md`。
- 注:逐样本是 N 次轻量 CPU 评测(N 通常小);若某 run 的 evaluate 很重,可选地让 evaluate.py 暴露 per-sample hook 优化,默认仍走子集调用以保持通用且与冻结指标一致。

### 5.2 方向发现(Claude 驱动 + 薄 schema 校验)
Claude 用 WebSearch/WebFetch 在 Arxiv / HF / paperswithcode 针对短板小结 + 研究方向检索;通用排序准则(对齐第一阶段 candidates):代码/权重/数据集可得性、与短板相关性、报告增益、实现 tier 成本。产出 `directions.json`(schema 见 §4)。`directions_schema.py` 仅校验/规范结构,**搜索本身是 Claude 的判断**。

### 5.3 实验执行(复用第一阶段脚本)
每方向一个 `exp_*/`,`variant.md` 写清改动。docker **默认复用主干镜像**;要加依赖就写 `FROM <主干镜像>` 的小 Dockerfile,`docker_env.py build` 出派生 tag。按 tier 实现:
- `config`:改主干推理参数(复用主干 infer 代码,换参数)
- `post-process`:在主干预测上加后处理步骤
- `pipeline`:串接额外组件/模型
- `train`:`train_launch.py` 产新权重,再用新权重推理
- `infer-tune`:步骤 2 闸门用的便宜档,本质是 config/post-process 的组合
`run_inference.py → predictions` → `compute_metrics → metrics.json`;逐实验 `progress.py` 记状态,重试上限 3。并发由 §6 的 dispatch 派发器按空闲卡分配,训练型排队。

### 5.4 `train_launch.py` — 通用训练壳(不是 trainer;真正 train 代码由 Claude 按论文/仓库写)
标准化四件最易错的事:
- **数据集获取**:下论文公开数据集(无则类似公开集)到 `caches/`,走 ModelScope>HF,记出处到 `exp_*/dataset_provenance.json`。
- **启动**:多卡(torchrun/accelerate)在派发到的卡上,输出 → `train.log`。
- **checkpoint**:写 `exp_*/ckpts/`,可从最新续。
- **预算 + 监控 + 恢复**:按方向配 time/step 预算,超则 kill 记 crash;中断可从最新 ckpt 续,`progress.py` 记 train 阶段。

## 6. `phase2_state.py` — 确定性受测核心(方案 C 的心脏)

独占读写 `state.json`,把最易错的记账钉成纯逻辑 + CLI(仿第一阶段 `progress.py`):

| 子命令 | 职责 |
|--------|------|
| `init` | 从 `results.tsv` 选 `status=ready` 最高分作初始主干,建 state.json |
| `backbone get` / `backbone promote` | 读当前主干 / 晋升(`version_n++`、移指针、记 history) |
| `round open` / `round record` | 开新轮(rNNN)/ 登记某 slot 的实验结果 |
| `gate check` | **纯函数**:候选相对当前主干提升 ≥ `PROMOTE_REL` 才返回 promote;边界单测 |
| `dispatch plan` | 输入=就绪实验+各自卡需求+一份 GPU 快照 → 输出分配;非训练优先、训练型卡紧时排队(**接收快照所以可测**) |
| `directions add` / `directions seen` | 跨轮已试方向登记与去重 |
| `resume` | 读 state 告诉 Claude 从 diagnose/infer-tune/search/score/promote 哪步续 |

### state.json schema(支持中断恢复)
```json
{
  "tag": "<run-tag>",
  "backbone": {"id": "p1:<name>", "source_dir": "models/<name>", "primary_metric": 0.0,
               "metrics": {}, "version_n": 0},
  "backbone_history": [{"version_n": 0, "id": "...", "primary_metric": 0.0, "promoted_at": "..."}],
  "inference_tuning": "pending",            // pending|explored|applied|skipped
  "round_counter": 0,
  "rounds": [{"id": "r001", "status": "scored",
              "slots": [{"slot": "a", "exp_dir": "rounds/r001/exp_a", "tier": "train",
                         "primary_metric": 0.0, "delta_pct": 0.0, "status": "discard"}]}],
  "directions_tried": ["<规范化方向指纹>"],
  "updated_at": "2026-05-28T..."
}
```

### 晋升门(明确决定)
- **相对 +5%**:`candidate.primary_metric >= backbone.primary_metric * (1 + PROMOTE_REL)`,`PROMOTE_REL=0.05`。
- 只认主指标,不加副指标守卫(本版从简,对齐"明显提升"措辞,避免噪声/随机种子带来的微小波动误晋升导致主干频繁抖动)。

## 7. 错误处理与边界

| 失败类型 | 处理层 | 策略 |
|---------|-------|------|
| GPU 全忙/卡不足 | `gpu_select.py` / `dispatch plan` | 训练型实验排队等卡;非训练型优先;长时间无卡则该轮先跑能跑的 |
| 镜像/依赖构建失败 | 实验执行 | 计入 retry_count,Claude 读日志改派生 Dockerfile |
| 数据集下载失败 | `train_launch.py` | 区分可重试(网络)与不可重试(权限/无公开数据);后者该方向放弃记 crash 并标因 |
| 训练 OOM | 阶段重试 | 预算内降 batch/精度;仍不行 → crash |
| 训练发散(NaN/到预算无改善) | `train_launch.py` | 记 crash,不阻塞同轮兄弟 |
| 推理崩溃 | 实验执行 | 有界重试 3 次 |
| `evaluate.py` 报错 | `compute_metrics.py` | 该实验评测失败 `status=crash`,不影响同轮其他 slot |
| 单 slot 崩溃 | 轮编排 | 该 slot=crash,另两个继续,该轮给存活者算分与晋升判定 |
| 搜索无可用结果 | 方向发现 | Claude 拓宽检索 / 进入"没主意"纪律(重读论文、组合近似命中、更激进改动),绝不卡死循环 |
| 整个 run 中断 | `phase2_state resume` | 读 state.json + 逐实验 progress.json + ckpts,从断点恢复:已 scored slot 跳过、已晋升主干不回退 |

**有界重试上限 3**(对齐第一阶段与 CLAUDE.md);crash 不阻塞链路;晋升只在显著提升时发生,避免噪声驱动的主干抖动。

## 8. 测试策略

Python 标准库 + pytest,镜像第一阶段布局(unit + contract + gpu-marked smoke)。脚本可独立测试;Claude 判断部分靠诊断报告 + results.tsv 留痕兜底。

### 单元测试(不碰真实 Docker/GPU)
| 脚本 | 测什么 | 怎么测 |
|------|--------|--------|
| `phase2_state.py` | 晋升门 +5% 边界、主干初始化、轮/实验登记往返、resume 决策、dispatch 训练排队、directions 去重、state.json 原子写 | tmp 目录建假 run + 假 results.tsv;dispatch 喂入 mock GPU 快照断言分配;门函数喂边界值 |
| `diagnose.py` | 子集视图构造、黑盒调用、最差 K、按字段分组、无分类字段降级 | 微型合成 dataset + predictions + 玩具 evaluate.py |
| `directions_schema.py` | directions.json 结构校验 | 合法/非法样例 |
| `train_launch.py` | 预算解析、ckpt 续选、数据出处记录 | mock 下载 + mock 启动,断言参数与 provenance |

### 契约测试(脚本间文件约定)
- 端到端 *dry-run*:玩具 dataset + 玩具 evaluate.py + stub docker/推理/训练,模拟一整轮文件流转(diag → directions → exp 预测 → metrics → results.tsv → state.json 晋升),断言每步产物 schema 与字段对得上。锁住跨脚本接口——最有价值的测试。

### 冒烟测试(需真实环境,默认 skip,`@pytest.mark.gpu`)
- 微型主干上跑通:一个 infer-tune 实验走 dispatch→infer→score→gate;一个微型 train 实验走 train_launch→ckpt→infer→score。

### 不测
- Claude 判断质量(短板解读、方向搜索与排序、实现代码)——靠诊断报告与人工可复核的 results/variant 留痕。
- 真实方向的提升幅度——agent 运行结果,非单元测试范畴。

**TDD 节奏**:每个脚本先写测试(红)→ 实现(绿)→ 重构,小步提交。

## 9. 交付物与 git 工作流

### 交付物
- `SKILL.md`(phase-2,独立第二个 skill,与现有 phase-1 SKILL.md 并列;phase-1 跑完产出主干后被调用)
- `references/optimization-loop.md`(优化循环细则,从 phase-2 SKILL.md 引用)
- 新脚本:`phase2_state.py` / `diagnose.py` / `train_launch.py` / `directions_schema.py`(薄)
- 复用:`gpu_select.py` / `docker_env.py` / `run_inference.py` / `compute_metrics.py` / `progress.py`
- tests/ 镜像第一阶段结构

### git 工作流
- **构建 skill 本身**:走 `GIT_CONVENTIONS.md` 的 `feature/phase2-optimization`。因 phase-2 复用 phase-1 脚本而 develop 尚无这些脚本,本分支从 `feature/phase1-reproduction` 切出(待 phase-1 合入 develop 后再随之整合);完成后按约定 squash 合回 develop。
- **skill 运行时的优化循环**:在专用分支 `optimize/<tag>` 上跑,每个实验产物直接 commit 留痕;keep/discard 由 `state.json` 主干指针管理,不用 git reset(见 §3 关键设计点)。
