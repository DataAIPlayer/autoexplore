# Phase-2 实操经验:gotchas + 通用 recipe

承接 SKILL.md 和 optimization-loop.md,集中收录**反复踩过的坑 + 通用 recipe**。
本 skill 是任务/数据/模型无关的:此处只保留普适规则与可复用代码片段,不出现任何具体任务、模型或框架名。

---

## 1. 训练 forward 优先复用官方/社区,反向工程是最后手段

**典型静默失败模式**:从推理管道反推训练 forward,跑得通、loss 在下降,但 loss 平台远高于该模型族的报告值,
或训完的适配器(全强度)把模型推坏;只有把适配器贡献缩到很小才接近 baseline——这几乎肯定是**漏件**,
不是噪声。

**最常见的漏件**(都是从推理代码看不出来的):
- **训练才传的额外条件输入**(模型生成时控制变量的旗标/计数/辅助图,推理时被外部 wrapper 默认填好)。
- **LoRA target_modules 默认集合**:许多复合架构(混合 MoE、双流注意力、modulation 等)的"该训"模块
  远不止 attention 的 `to_q/k/v/out.0`,还包括跨流 projection、各种 MLP、modulation 层等。
- **训练 vs 推理的文本/conditioning 分布失配**:训练用空 prompt、推理却由内部 VL 自动 caption,跨注意力分布完全不同。
- **多帧/视频 VAE 的时间维处理**:把多帧拼一起编码会被时间压缩,逐帧 `f=1` 编码才能保留帧数,否则下游 packing 形状对不上。

**搜索顺序(找到一个就停)**:
1. **模型作者 repo 的 `examples/`**(`<repo>/examples/training/`, `<repo>/training/`, `<repo>/scripts/train_*.py`)。
2. **支持该模型族的成熟训练框架**(通用扩散/LM 训练库),GitHub 搜 `train <model_family>` / `finetune <model>`。
3. **官方文档/blog/示例**中的 fine-tuning 章节。
4. **该模型 GitHub issues** 里 `fine-tune` / `training` / `LoRA` 主题线程,常有人贴成功配置或失败诊断。

**过拟合自检(forward 正确性 gate)** —— 投入大规模训练之前必跑:
```
单样本,百步级别训练,grad clip 1.0,合理学习率。
loss 必须可见下降到显著低于"模型平均/常数预测"的 baseline。
若 loss 在某高水位震荡不掉,forward 必然有漏件——别继续训,回去对比官方代码。
```

---

## 2. 多卡分发 recipe(训练 + 推理)

**入口都是 `gpu_select.py`**(动态查 nvidia-smi 给出当前空闲卡列表;每次启动新任务前重抓一次)。
卡数 N 由"满足 `--min-free-mib` 的空闲卡数量 ∩ 你想用的 `--count`"决定,自动退化为单卡。

### 训练:DDP via accelerate(或同等启动器)
```bash
GPUS=$(uv run scripts/gpu_select.py --count <max_N> --min-free-mib <model_mem_floor>)
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
docker run --rm --runtime=nvidia --gpus "device=$GPUS" \
    --user $UID:$GID -v ... -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    <image> bash -c "accelerate launch --num_processes $N <train.py> ..."
```
要点:
- `--num_processes` 必须 = 暴露给容器的卡数;别混搭(暴露 8 但 N=4)。
- 多数训练库的 progress bar 显示**总步数**,墙钟≈总步数 × per-step / N。
- LoRA batch=1 时 DDP 主要省时间,不直接增加模型容量;想增大 effective batch 用 `--gradient_accumulation_steps`。
- **显存 race**:多租户机器上,`pipe.to('cuda')` 加载阶段 vs 别人抢内存会 OOM。一是给 `--min-free-mib` 充裕余量
  (模型显存 + ~10% headroom),二是 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 推理:数据并行 sharding
```bash
GPUS=$(uv run scripts/gpu_select.py --count 4 --min-free-mib <model_mem_floor>)
IFS=',' read -ra G <<< "$GPUS"
mkdir -p $EXP/predictions
for i in $(seq 0 $(( ${#G[@]} - 1 ))); do
  GPU=${G[$i]} SHARD=${i}/${#G[@]} EXP_REL=$EXP \
    bash <inference-launcher>.sh > $EXP/shard${i}.log 2>&1 &
done
wait
# 合并:
cat $EXP/predictions/predictions.shard*.jsonl > $EXP/predictions/predictions.jsonl
```
推理脚本需读 `--shard N/M`,跳过 `idx % M != N` 的样本。每张卡分到 ≥1 个样本就值得分(模型加载是大头,被并行了)。
每个 docker 各自挂 cache(`:ro`),`--user $UID:$GID --runtime=nvidia` 一个都不能少。

---

## 3. 网络文件系统缓存死锁

**症状**:模型/数据下载或加载长时间停顿,日志反复出现 `Still waiting to acquire lock on .../<repo>` 或
`OSError(37, 'No locks available')`。

**根因**:很多模型加载/下载工具用 `fcntl.flock` 对单个文件加锁(防并发写)。NFS、SMB 等网络文件系统在
未启用 lockd 或不支持 BSD flock 时会返回 ENOLCK,工具陷入重试循环。

**修法**(首次使用前):
```bash
mkdir -p /tmp/<cache_name>_local
rsync -a <host_nfs_cache>/ /tmp/<cache_name>_local/   # 复制到本地 FS;SSD 上典型 ~1 min/GB
```
docker mount 改成 `-v /tmp/<cache_name>_local:/cache/<name>`(不再带 `:ro`,因为缓存框架要写 `.lock`)。
新下载也优先指向本地 cache(`HF_HOME=/tmp/...` 或同等环境变量),完成后视情况备份回 NFS。

**判别本机 cache 路径是不是 NFS**:`stat -f -c %T <path>` 看类型,或 `findmnt -T <path>` 看挂载源。
开始前判别一次,省一次几十分钟的卡顿。

---

## 4. 单向单调后处理 = 主干本身

如果发现一个**确定性、对主指标单调非降**的后处理(典型例:过滤明显退化的预测项、补全显式缺失项、
deterministic mask cleanup 等),首轮验证一次单调性后**默认开启**:
- 主干"报告分"= `model 输出 + 默认后处理`,而不是裸输出。
- `diagnose.py` 用的 predictions 也是后处理过的版本——不然短板诊断会偏向"模型尚未做的清理"。
- 它**不**算独立 infer-tune 实验,不消耗 `directions.json` 槽位。

判定单调性的小检查:
```
在前 1-2 个样本上,后处理后的每个样本主指标 ≥ 后处理前。
若有任一样本下降,说明它不是单向单调,要么调参数(让单调成立),要么作为可选实验。
```

---

## 5. peft LoRA state-dict 名字往返

**症状**:`set_peft_model_state_dict(model, sd)` 返回的 `missing_keys` 计数等于 LoRA 参数总数,
`unexpected_keys=0`;推理跑得通但效果 = baseline。

**根因**:`get_peft_model_state_dict` 保存时**剥掉了 adapter namespace**(如 `.default.`,为了跨命名空间可移植);
而 `set_peft_model_state_dict` 在部分 peft 版本下不会自动加回去,model.state_dict() 里却需要这层。

**手动 remap(可靠 fallback)**:
```python
import torch
sd = torch.load(lora_path, map_location="cuda")
remapped = {}
for k, v in sd.items():
    if "lora_A.weight" in k:
        remapped[k.replace("lora_A.weight", "lora_A.default.weight")] = v
    elif "lora_B.weight" in k:
        remapped[k.replace("lora_B.weight", "lora_B.default.weight")] = v
    else:
        remapped[k] = v
res = model.load_state_dict(remapped, strict=False)
n_missing_lora = sum(1 for k in res.missing_keys if "lora_" in k)
assert n_missing_lora == 0, f"{n_missing_lora} LoRA keys still missing — LoRA NOT applied"
```

**Sanity check(强烈推荐)**:加载完后跑 1 个样本,对比 LoRA 输出 vs baseline 输出在像素/张量层面应该**明显非零差异**
(典型 ≥10% 平均差异)。若几乎等于 0,说明 LoRA 没生效——别浪费几小时跑全量推理,先 debug 适配器装载。

部分集成训练框架(用自己封装的 `load_lora` API)内部已处理这条坑;但只要你跨框架/版本搬运 ckpt,
都应该手动做上面的 sanity check。

---

## 6. 容器内符号链接必须用相对路径

**症状**:容器里跑脚本读软链报 `FileNotFoundError`,但主机上 `ls -L` 文件明明存在。

**根因**:`os.symlink((src/iid).resolve(), link)` 写入的是**主机绝对路径**,容器只挂了部分主机路径
(典型只挂了 `/work` 或类似容器内根),主机上其他路径在容器里根本不存在,软链立刻悬空。

**修法**:数据子集/视图统一用相对路径:
```python
import os
from pathlib import Path
link = subset_dir / sample_id
if not link.is_symlink():
    rel = os.path.relpath((src_dir / sample_id).resolve(), subset_dir.resolve())
    os.symlink(rel, link)
```
只要相对结构在主机和容器里一致,相对路径软链两边都能解析。

**最容易漏的地方**:把"主干 predictions"以 symlink 形式喂给下游脚本(例如 post-process 输出目录、
recursive/refinement 实验的 base_dir 等)——确保这些脚本写的 symlink 也是相对路径。

---

## 7. 杠杆叠加 ≈ 加法,但"组合所有 winner"可能反而掉

**经验法则**:每个独立 +X% 杠杆,叠加后的总增益**通常 ≈ 各 X% 之和**(不是乘积)——因为每个杠杆
往往修复**不同子集**的弱点,效果近似独立。

**但**:**激进的"all in"组合容易回退**——把"更大模型容量 + 更多数据 + 更多步数"一次性堆上,
常因优化难度非线性上升(更多自由度需要更精细的 lr / warmup / 早停)反而下降。

**实操**:
- 发现新 winner,**先单独叠加最近 promoted 主干**测试;过门即晋升。
- 想合并训练时,先在小子集上 sanity check loss 仍能下降到对应平台。
- 同方向上别一次升 ≥2 个维度(rank、数据量、训练步数),否则你不知道是哪个维度让它退化。

---

## 8. 天花板信号 & 升级触发

**触发信号**(任一即可):
- 同一 tier 内 ≥3 个独立实验,主指标在 ±1% 同一带波动;
- 单一杠杆做 3 个变体(步数/rank/数据量)出现明显 U 型(中段最优,两边都掉)→ 已饱和;
- 加 1 倍数据 / 加 1 倍 rank / 加 1 倍步数,收益 <1%。

**触发动作**(从这一 tier 跳到上一 tier):
- `infer-tune` 饱和 → `pipeline`(组件级:子任务串接、多模型协作、自适应路由);
- `pipeline` 饱和 → `train`(适配器微调);
- 适配器微调饱和 → 升级到更大的训练范式(全微调 / 更大模型 / 重新做数据 / 换基线模型)。

**何时回报给用户**:连撞两次"all-in 也不过门"——这通常意味着当前预算下确实到了结构性天花板。
此时应给用户一个**完整状态报告 + 选项清单**(继续 N 倍预算 / 接受现状 / 换大方向),由用户决定是否再投。
这不违反"绝不停下问人"——那条针对的是循环内部"做不做下一个"的决策,不针对"换大方向"这种战略转向。

---

## 9. 主干升级时的诊断习惯

每次 `promote_backbone` 后必跑 `diagnose.py`,**对比上一版的诊断 md**。常见模式:
- 训练型晋升时,改善往往集中在**某一类样本**(例如低/中难度组),另一类样本(高难度)**几乎不变**;
- 持续训练晋升时,改善逐步从"易"扩到"难",中间会出现**改善带"漂移"**的现象——这是判断继续训练 vs 换 tier 的好信号;
- 某个子组分数对训练**不再敏感**,说明剩下的瓶颈是**结构性**(固定输出维度、固定模板、词表限制等),
  需要 pipeline 杠杆或换更大输出空间的模型,而不是再训。

诊断结果直接驱动下一轮的 `directions.json` 选向——别把诊断当走过场。

---

## 10. 常用模板

下面是几个反复用得到的代码片段,直接抄进新 run。

**容器加载防 OOM**:
```bash
docker run --rm --runtime=nvidia --gpus device=$GPU --user $UID:$GID \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v ... \
    <image> bash -c "..."
```

**大文件下载带"卡死自动断点重连"**(代理不稳定时):
```bash
curl -sS -C - -L --connect-timeout 30 --speed-time 60 --speed-limit 2048 \
     -x "$HTTP_PROXY" -o "$OUT" "$URL"
# loop until $OUT 是合法文件(用对应格式的校验,如 parquet footer / safetensors header)
```
关键是 `--speed-time 60 --speed-limit 2048`:连续 60s 速度 <2KB/s 就 abort,外层循环再 `-C -` 续传。

**训练监控**:大多数训练入口默认不打 loss 到 stdout(只打 tqdm 进度),需显式开 tensorboard/wandb/swanlab 之类的
日志后端;否则 loss 曲线只能在事后从 ckpt 推算。

**模型重启的 LoRA 续训**:区分两条路径——
- "Full-model resume"(`--resume_from_checkpoint` 之类):期望 ckpt 包含全部模型权重,LoRA-only ckpt 会 unexpected keys 报错。
- "LoRA-only resume"(`--lora_checkpoint` 之类):专用入口,只装 LoRA 权重到当前已构建的适配器上。
两者别混用;读训练库参数解析时确认这两类是哪一个。
