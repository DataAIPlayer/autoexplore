---
name: autoexplore-phase2
description: Use after phase-1 produced a baseline — drives phase-2 iterative optimization: set the chosen model as the backbone, diagnose its test-set weaknesses, try cheap inference-pipeline tuning first, then run a never-ending search loop (Arxiv/HF/paperswithcode → 3 directions → train if needed → score → promote on relative +5%).
---

# autoexplore 第二阶段:模型效果迭代优化

输入:第一阶段在 `runs/<tag>/` 产出的冻结 `dataset/`、`evaluate.py`、`results.tsv` 与 ≥1 个 `status=ready` 模型。
产出:在同方向上不终止地优化"模型主干",每次有明显提升(相对 +5%)就晋升新主干,直到人工终止。

环境:云 GPU 服务器,Docker + NVIDIA runtime,多卡;**复用第一阶段主干的 docker 镜像**;按空闲度选卡。
**多卡可用就用**:训练默认走 DDP(`accelerate launch --num_processes N` 或同等启动器),
推理默认按 `--shard N/M` 数据并行——两者都用 `gpu_select.py` 算出来的当前空闲卡集合自动扩展,
卡少自动退化为单卡(详见 [references/optimization-loop.md](references/optimization-loop.md) 步骤 3.b)。

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

**训练方向铁律**:实现前先翻官方/社区训练脚本(模型作者 repo 的 `examples/`、HF blog/示例、模型的 GitHub issues 里
"how to fine-tune"线程、成熟的多模型训练框架)——命中就直接复用,**不要靠读推理管道凑训练 forward**。
推理路径常常隐藏训练才需要的额外输入和 LoRA target 集合,反向工程极易踩到"看似在训、loss 在降、但学的方向是错的"
这一类静默失败。具体清单、典型坑、过拟合自检方法见 [references/lessons.md](references/lessons.md)。

完整细则见 [references/optimization-loop.md](references/optimization-loop.md)。

## 关键纪律
- `dataset/` 与 `evaluate.py` 全程**不可变**,保证主干各版本与实验可比。
- **便宜档先行**:推理管道调优闸门先于昂贵搜索;搜索内 config/post-process 优先于训练。
- **晋升只认主指标相对 +5%**,避免噪声/随机种子的微小波动误晋升致主干抖动。
- **单向单调后处理算主干本身**:若有"确定性、对主指标单调非降"的后处理(例如过滤明显退化的预测项),
  在首轮验证一次单调性,之后**默认应用**——它不是独立 infer-tune 实验,而是主干"报告分"的一部分,
  诊断/晋升用的预测路径都包含这一步。
- **keep/discard 用 state.json 主干指针,不用 git reset**:失败实验目录保留作研究档案。
- 容器/训练输出进 log,只失败时 `tail`;逐实验重试上限 3,crash 不阻塞同轮兄弟。
- 中断可恢复:入口 `resume` 读 state.json;已 scored slot 跳过、已晋升主干不回退。
- **容器纪律(继承第一阶段)**:每次 `docker run` 带 `--user $UID:$GID --runtime=nvidia`;
  caches 以 `:ro` 挂 `/cache/{modelscope,huggingface,torch}`,env 注入对应 `*_CACHE`/`*_HOME`。
  **若主机 cache 在网络文件系统**(NFS/SMB 等不支持 `fcntl.flock` 的盘),先 `rsync` 到本地盘
  (`/tmp/<cache>_local` 或同等本地 FS,支持 flock)再挂——多数模型下载/缓存工具会 flock 单文件,
  在 NFS 上 ENOLCK 会让加载卡住几十分钟而不报错。
- **容器内符号链接用相对路径**:跨主机 ↔ 容器路径挂载时,绝对路径 symlink 在容器里通常指向不存在的位置,
  导致 `FileNotFoundError`;构造数据子集/视图时一律用 `os.path.relpath` 写软链。
- **晋升后 commit 留痕仅记录小档案**:`results.tsv`、`state.json`、`diagnostics/`、脚本本身;**不 commit**
  权重 ckpt、训练样本图、推理输出图(几百 MB ~ GB 级,属于本地工件,本就该被项目 `.gitignore` 兜住)。
- **天花板信号 & 升级触发**:同一 tier 内 2-3 个独立实验都收敛到主指标 ±1% 同一带,就别再加同类实验,
  立刻升一级(config→pipeline→train→更大数据/全微调/换架构),否则就是在烧 GPU 抖动随机种子。
- **晋升不许 test-snoop**:推理超参(LoRA scale、steps、N 槽数等)若只能"事后在测试集挑出来",
  视同泄漏(本质等于"用测试集挑超参");锁定训练时的固定超参,把那一组结果当主干报告。

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/phase2_state.py {init,resume,gate,dispatch,backbone-get}` | 状态/晋升门/派发(确定性核心) |
| `scripts/diagnose.py` | 黑盒调用 evaluate.py 出短板分解 |
| `scripts/directions_schema.py --file <directions.json>` | 方向 schema 校验 |
| `scripts/train_launch.py` | 数据出处/多卡启动/ckpt 续/预算 |
| `scripts/{gpu_select,docker_env,run_inference,compute_metrics,progress}.py` | 复用第一阶段 |
