# 优化循环细则(第二阶段)

承接 SKILL.md。所有路径相对 repo 根;命令均从 repo 根 `uv run`。
状态唯一真相源 = `runs/<tag>/phase2/state.json`,只经 `scripts/phase2_state.py` 读写。
进入任意阶段前先 `resume` 决定从哪步续:

```bash
uv run scripts/phase2_state.py resume --run-dir runs/<tag>
# → {"action": "init|diagnose|infer_tune|search|execute|promote_check"}
```

## 步骤 0:Setup(唯一人工关卡)
1. 续用第一阶段 `<tag>`;`git checkout -b optimize/<tag>`(运行时专用分支)。
2. `uv run scripts/phase2_state.py init --run-dir runs/<tag> --tag <tag>`
   → 主干 = `results.tsv` 中 `status=ready` 且 primary 最高者。
3. 校验 `dataset/` 与 `evaluate.py` 在位且自第一阶段冻结后未变。确认后进入自主循环。

## 步骤 1:短板诊断(每次主干变化都重跑)
对当前主干已产出的 `predictions.jsonl`(初始主干来自第一阶段 `models/<name>/predictions.jsonl`):
```bash
uv run scripts/diagnose.py \
  --predictions <主干 predictions.jsonl> \
  --dataset runs/<tag>/dataset --evaluate-py runs/<tag>/evaluate.py \
  --out-json runs/<tag>/phase2/diagnostics/diag_<backbone_id>.json \
  --out-md   runs/<tag>/phase2/diagnostics/diag_<backbone_id>.md --worst-k 10
```
读 `.md`,把短板小结(哪些分组/样本/副指标最差)记进同名 `.md` 末尾,作为搜方向输入。
诊断后调用 `mark_diagnosed`(经脚本函数或后续封装)推进 `last_diagnosed_version`。

## 步骤 2:推理管道调优闸门(便宜档,先于训练)
判断:仅调主干推理管道(参数/后处理,无新方法、不训练)是否可能改善上面的短板?
- 有空间 → 按"实验执行"跑几个 `tier=infer-tune` 实验(config/post-process 组合);
  任一过门(见下)→ 晋升主干 → 回步骤 1。
- 无空间或都没过 → `set_inference_tuning explored` → 进步骤 3。

纪律:便宜档永远先穷尽,再动昂贵搜索。**确定性单调后处理(对主指标单调非降的过滤/清理)算主干本身**,
不当 infer-tune 实验单列;首轮验证一次单调性后默认开启,诊断与评分用的预测路径都走它。
其他独立 infer-tune 候选:推理步数、固定输出 N、分辨率桶、CFG/guidance、对预测的形态学清理等。
经验:这些独立 config 杠杆单杠杆的提升通常在个位数 % 量级,**前 2-3 个变体若收敛到主指标 ±1% 同一带,
就 `explored` 不要再加同类**(详见 [lessons.md](lessons.md) 的"天花板信号 & 升级触发")。

## 步骤 3:不终止搜索优化循环
LOOP(直到人工终止):

### a. 搜方向(agent 判断)
在 Arxiv / HF / paperswithcode 针对短板检索;通用排序:代码/权重/数据集可得性、与短板相关性、报告增益、tier 成本。选 3 个,写 `runs/<tag>/phase2/rounds/<rid>/directions.json`,搜索过程进 `search.log`。校验并去重:
```bash
uv run scripts/directions_schema.py --file runs/<tag>/phase2/rounds/<rid>/directions.json
# 对每个方向 directions_seen 跳过已试(经 phase2_state 函数);open_round 会登记去重
```

**`tier=train` 方向必做前置检索**:
1. 先翻**模型官方 repo 的 `examples/`**(`<repo>/examples/training/`, `<repo>/training/`, `<repo>/scripts/train_*.py`),命中即用。
2. 再翻**支持该模型族的成熟训练框架**(各家通用扩散/语言模型训练库、HuggingFace `examples/`、社区 LoRA 训练器),搜 `train <model>` / `finetune <model>` / `LoRA <model>`。
3. 翻**该模型 GitHub issues 里"how to fine-tune / training / LoRA"线程**,常见有人贴成功配置或失败诊断。
4. **只有以上都没命中**才反向工程推理管道写 train.py;此时务必先做"**过拟合自检**":
   单样本训百步级别,loss 必须可见下降到显著低于"模型默认预测"的 baseline,**否则 forward 必然漏件**——
   常见漏件包括训练才需要的额外条件输入、模型族特有的 LoRA target 模块集合、与推理不一致的文本/conditioning 分布、
   多帧 VAE 的时间/空间压缩特性等(详见 [lessons.md](lessons.md))。

### b. 派发并执行 3 个实验(按空闲卡并发,训练型排队 + 多卡扩展)
```bash
GPUS=$(uv run scripts/gpu_select.py --count 8 --min-free-mib 20000 | tr ',' '\n')
uv run scripts/phase2_state.py dispatch \
  --experiments '[{"slot":"a","needs_gpus":1,"is_training":false}, ...]' \
  --free-gpus "$(echo $GPUS | tr ' ' ',')"
# → {"assigned": {"a":[0],...}, "queued": ["c"]}  训练型卡紧时进 queued,等卡再跑
```

每个 `exp_<slot>/` 按 tier 实现并产出 `predictions.jsonl`:
- **复用主干 docker 镜像**;需额外依赖才写 `FROM <主干镜像>` 的小 Dockerfile + `docker_env.py build`。
- `config`:换主干推理参数 · `post-process`:在主干预测上加后处理 · `pipeline`:串接组件 ·
  `train`:`train_launch.py` 取数据→多卡训→ckpt→用新权重推理 · `infer-tune`:config/post-process 组合。
- 推理:`run_inference.py`(复用第一阶段,挂 caches 与 dataset,带 `--user`/`--runtime nvidia`)。
- 评分:`compute_metrics.py --evaluate-py runs/<tag>/evaluate.py --predictions <exp>/predictions.jsonl --dataset runs/<tag>/dataset --out <exp>/metrics.json`。
- 逐实验 `progress.py` 记 stage,重试上限 3;crash 记一行不阻塞兄弟。
全部产出后 `record_slot`(done/crash),凑齐则轮转 `scored`。

**多卡使用默认开启**(卡少自动退化为单卡):`gpu_select.py` 给空闲清单,实验脚本据此扩展:

- **训练(DDP)**:训练入口若是 `accelerate launch`(主流扩散/LM 训练库都是),N 张可用卡就
  `accelerate launch --num_processes N <train.py> ...`,进度条显示总步数会自动按进程数除。
  注意 LoRA 常用 batch_size=1,DDP 主要省 wall-time(等价于扩大 effective batch),并不直接增加模型容量;
  想增大 effective batch 改 `--gradient_accumulation_steps`。显存 race 风险:每张卡 `pipe.to('cuda')` 时若
  另一租户抢内存会 OOM,加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,并按模型实际显存留充足
  headroom(`gpu_select --min-free-mib <N>`)。

- **推理(sharding)**:把数据集按 `--shard N/M` 切分,每个 docker 跑一个 shard。模板:
  ```bash
  GPUS=$(uv run scripts/gpu_select.py --count 4 --min-free-mib 60000); IFS=',' read -ra G <<< "$GPUS"
  for i in 0 1 2 3; do
    GPU=${G[$i]} SHARD=${i}/4 EXP_REL=... bash <launcher>.sh > <exp>/shard${i}.log 2>&1 &
  done
  wait
  # 跑完合并:cat <exp>/predictions/predictions.shard*.jsonl > <exp>/predictions/predictions.jsonl
  ```
  每张卡分到 ≥1 个样本就值得分(模型加载是大头,被并行;采样阶段也被并行)。
  每个 docker 也要 `--user $UID:$GID --runtime=nvidia`,各自挂 cache(`:ro`)。

- **GPU 缓存放置**:主机 cache 在网络文件系统(不支持 `fcntl.flock`)时,模型加载/下载会因 ENOLCK
  卡死几十分钟。**首次使用前**:
  ```bash
  rsync -a <host_nfs_cache>/ /tmp/<cache>_local/      # 复制到本地 FS,支持 flock
  # docker 挂 -v /tmp/<cache>_local:/cache/<name>     # 而不是 NFS 路径
  ```
  下载新模型/数据集也优先指向本地 cache 路径(`HF_HOME=/tmp/...` 或同等环境变量),完成后再视情况备份回 NFS。

### c–d. 算分与晋升
```bash
uv run scripts/phase2_state.py gate --candidate <最佳 exp primary> --backbone <主干 primary>
# {"promote": true}  当且仅当相对 ≥ +5%
```
- 过门 → `promote_backbone`(version_n++、移指针、重置 inference_tuning=pending)→ `git add <exp> && git commit` 留痕 → `close_round` → 回步骤 1(螺旋)。
- 不过门 → 三者各记 `results.tsv` 一行(`discard`/`crash`)→ `close_round` → 回 a 选新方向。

### e. 永不停下
绝不问人是否继续。"没主意"时:重读论文、组合差一点的近似命中、试更激进改动——直到人工终止。

## 日志与重试纪律
- 容器/训练输出进 `run.log`/`train.log`,只失败时 `tail -n 50`,绝不全量进上下文。
- 逐实验重试上限 3;crash 不阻塞同轮兄弟,也不停整循环。
- keep/discard 由 state.json 主干指针管理,**不用 git reset** 抹实验目录(失败也是研究档案)。
