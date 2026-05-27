# 复现循环细则(单个候选模型)

对 `runs/<tag>/candidates.json` 中每个候选,按 priority 升序,在 `runs/<tag>/models/<name>/` 工作目录内执行。开始前读 `progress.json`,若 `stage` 为 `ready`/`crash` 则跳过,否则从记录的 stage 恢复。

## 阶段 A:准备环境

1. 选卡(每次进入都重查,因为别的进程可能占卡):
   ```bash
   GPUS=$(uv run scripts/gpu_select.py --count 1 --min-free-mib 20000)
   ```
   非零退出表示无空闲卡——决定等待或减小 --count / --min-free-mib。
2. 写一份 `Dockerfile`(参照模型仓库 README/requirements,由 Claude 生成)放进 `models/<name>/`。
   - 基础镜像若已含 torch+transformers(host 上 `docker images` 看),复用为 base 比从 `pytorch/pytorch:*` 起跑能省 6 GB+ 拉取。
   - 容器内 `pip install` 在某些网络下被节流到 ~10 kB/s。**最稳的做法**是 host 上预取 wheels(走 `~/.claude/projects/.../memory/network_sources.md` 里的 TUNA 直连)→ `COPY wheels /tmp/wheels` → `RUN pip install --no-deps --find-links /tmp/wheels …`,绕过容器内网络。
   - 某些基础镜像残留 legacy `.egg` 包(如 `huggingface_hub-0.35.0rc0-py3.11.egg`),pip 不会卸,但 import 时优先;装新包前用 `find <site-packages> -maxdepth 1 -name '<pkg>-*.egg' -exec rm -rf {} +` + 清 `easy-install.pth` 对应行。
3. 构建/复用镜像:
   ```bash
   uv run scripts/docker_env.py build --tag autoexplore/<name> --dockerfile-dir runs/<tag>/models/<name>
   ```
   输出 `reused` 表示已存在跳过(对齐"已构建过则忽略")。若 host 的 docker daemon 带 `HTTP_PROXY=127.0.0.1:*` 误伤容器内 pip(每个文件 10 kB/s),改用 `docker build --network=host ...` 直接绕过。
4. 记录进度:
   ```bash
   uv run scripts/progress.py set --model-dir runs/<tag>/models/<name> \
     --model <name> --stage A --gpus "$GPUS" --image-tag autoexplore/<name>
   ```

## 阶段 B:下载

5. **首选:host 端预取权重**(走 ModelScope/TUNA,见 [[network-sources]] memory),落 `<repo_root>/caches/{modelscope,huggingface,torch}`,**不落 `~/.cache/`**——项目自包含,跨 host 复用,与系统盘量解耦。预取脚本写法举例:
   ```python
   from modelscope import snapshot_download
   snapshot_download("Qwen/Qwen-Image-Layered", cache_dir="caches/modelscope")
   ```
   ```python
   from huggingface_hub import snapshot_download as hf_snap
   hf_snap("cyberagent/layerd-birefnet", cache_dir="caches/huggingface")
   ```
   仅当模型只在 HF 才用 huggingface_hub;**记得**在 import 前 `os.environ.pop("ALL_PROXY", None)`,否则 httpx 会因 socks5 报 ImportError(见 [[network-proxy]])。
6. 容器内若仍需联网下载小补丁(比如 torch.hub fallback 的 `big-lama.pt`),通过 `docker_env.py run` 起容器,输出进 run.log:
   ```bash
   uv run scripts/docker_env.py run --tag autoexplore/<name> --gpus "$GPUS" \
     --mount "$(pwd)/runs/<tag>/models/<name>:/work" \
     --mount "$(pwd)/caches/huggingface:/cache/huggingface:ro" \
     --env HF_HOME=/cache/huggingface --env HF_MODULES_CACHE=/tmp/hf_modules \
     --user "$(id -u):$(id -g)" --runtime nvidia \
     --log runs/<tag>/models/<name>/run.log \
     --inner-cmd "<下载命令>"
   ```
7. 失败转阶段 D;成功 `progress.py set ... --stage B`。

## 阶段 C:验证推理

8. 有官方示例代码用示例;没有则 Claude 自建最小推理脚本,放进 `models/<name>/`。推理脚本约定把每层写到 `/work/predictions/<image_id>/L*.png`,jsonl 写到 `/work/predictions.jsonl`(`alpha` 字段用相对路径 `predictions/<image_id>/L*.png`)。
9. 跑推理验证(对 dataset 产出 predictions.jsonl 到 /work/predictions.jsonl):
   ```bash
   REPO_ROOT=$(pwd)
   uv run scripts/run_inference.py --tag autoexplore/<name> --gpus "$GPUS" \
     --run-dir runs/<tag>/models/<name> \
     --dataset-dir runs/<tag>/dataset \
     --extra-mount "$REPO_ROOT/caches/modelscope:/cache/modelscope:ro" \
     --extra-mount "$REPO_ROOT/caches/huggingface:/cache/huggingface:ro" \
     --extra-mount "$REPO_ROOT/caches/torch:/cache/torch:ro" \
     --env MODELSCOPE_CACHE=/cache/modelscope \
     --env HF_HOME=/cache/huggingface \
     --env TORCH_HOME=/cache/torch \
     --env HF_MODULES_CACHE=/tmp/hf_modules \
     --user "$(id -u):$(id -g)" --runtime nvidia \
     --log runs/<tag>/models/<name>/run.log \
     --infer-cmd "<推理命令,输出写到 /work/predictions.jsonl>"
   ```
   `--dataset-dir` 显式把 host 上的 `runs/<tag>/dataset` 挂到容器内 `/work/dataset`(`run_dir` 默认只能看到模型自己的目录,看不到 dataset)。
10. 成功 → `progress.py set ... --stage ready` 并把推理参数也记上(见阶段 D);循环结束。失败 → 阶段 D。

### 加速:单模型 N-GPU 分片

模型间仍串行,但**单模型内**当 smoke 1 样本 > 60 s 时,把 dataset 切 N 片并发(本次 Qwen-Image-Layered 这样把 90 张 16 h 压到 2 h):

- infer.py 接受 `--shard i/N --skip-existing`,只处理 `idx % N == i` 的样本,已存在的跳过。
- wrapper 同时启动 N 个 `docker run`(每个一张空闲卡),全部结束后 `cat predictions.shard*.jsonl > predictions.jsonl`。
- N 选 ≤ 空闲卡数;每卡显存 ≥ 模型权重 + 推理活动量(Qwen-Image-Layered 在 bf16 res=640 用 ~35 GB)。

## 阶段 D:有界重试

11. `retry_count += 1` 写进 progress.json;同步用 `--infer-config K=V` 把当前推理参数(steps/resolution/layers/shards 等)记上,下次复盘"是配置变了还是网络抖了"对得上账:
    ```bash
    uv run scripts/progress.py set --model-dir runs/<tag>/models/<name> \
      --model <name> --stage A --gpus "$GPUS" --image-tag autoexplore/<name> \
      --retry-count 1 \
      --infer-config steps=30 --infer-config resolution=640 \
      --infer-config layers=8 --infer-config shards=3
    ```
12. 读 `tail -n 50 run.log` 的错误:
    - 简单可修(缺依赖/路径/CUDA 版本/缺 `--runtime=nvidia` 或 `--user`):改 Dockerfile 或 run 参数,回阶段 A。
    - 根本性失败(架构不兼容/资源不够):放弃。
13. `retry_count >= 3` → 放弃,`progress.py set ... --stage crash`,在 results.tsv 记 crash:
    ```bash
    uv run scripts/progress.py result --run-dir runs/<tag> --model <name> \
      --primary-metric 0.0 --memory-gb 0.0 --status crash --description "<原因>"
    ```

## 成功后:推理 + 评测(步骤 9–10)

阶段 C 已产出 predictions.jsonl,计算指标:
```bash
uv run scripts/compute_metrics.py --evaluate-py runs/<tag>/evaluate.py \
  --predictions runs/<tag>/models/<name>/predictions.jsonl \
  --dataset runs/<tag>/dataset \
  --out runs/<tag>/models/<name>/metrics.json
```
记录结果:
```bash
uv run scripts/progress.py result --run-dir runs/<tag> --model <name> \
  --primary-metric <值> --memory-gb <值> --status ready --description "<复现说明>"
```

## 日志纪律

- 容器输出一律进 run.log,只在失败时 `tail -n 50 run.log`,绝不让全量日志进上下文。
- 重试上限 3 次(对齐 CLAUDE.md "3 次后停下重新评估")。
