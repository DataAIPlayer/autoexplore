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
纪律:便宜档永远先穷尽,再动昂贵搜索。

## 步骤 3:不终止搜索优化循环
LOOP(直到人工终止):

### a. 搜方向(agent 判断)
在 Arxiv / HF / paperswithcode 针对短板检索;通用排序:代码/权重/数据集可得性、与短板相关性、报告增益、tier 成本。选 3 个,写 `runs/<tag>/phase2/rounds/<rid>/directions.json`,搜索过程进 `search.log`。校验并去重:
```bash
uv run scripts/directions_schema.py --file runs/<tag>/phase2/rounds/<rid>/directions.json
# 对每个方向 directions_seen 跳过已试(经 phase2_state 函数);open_round 会登记去重
```

### b. 派发并执行 3 个实验(按空闲卡并发,训练型排队)
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
