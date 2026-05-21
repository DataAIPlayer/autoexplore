# 复现循环细则(单个候选模型)

对 `runs/<tag>/candidates.json` 中每个候选,按 priority 升序,在 `runs/<tag>/models/<name>/` 工作目录内执行。开始前读 `progress.json`,若 `stage` 为 `ready`/`crash` 则跳过,否则从记录的 stage 恢复。

## 阶段 A:准备环境

1. 选卡(每次进入都重查,因为别的进程可能占卡):
   ```bash
   GPUS=$(uv run scripts/gpu_select.py --count 1 --min-free-mib 20000)
   ```
   非零退出表示无空闲卡——决定等待或减小 --count / --min-free-mib。
2. 写一份 `Dockerfile`(参照模型仓库 README/requirements,由 Claude 生成)放进 `models/<name>/`。
3. 构建/复用镜像:
   ```bash
   uv run scripts/docker_env.py build --tag autoexplore/<name> --dockerfile-dir runs/<tag>/models/<name>
   ```
   输出 `reused` 表示已存在跳过(对齐"已构建过则忽略")。
4. 记录进度:
   ```bash
   uv run scripts/progress.py set --model-dir runs/<tag>/models/<name> \
     --model <name> --stage A --gpus "$GPUS" --image-tag autoexplore/<name>
   ```

## 阶段 B:下载

5. 在容器内拉安装包 + 模型权重,输出进 run.log:
   ```bash
   uv run scripts/docker_env.py run --tag autoexplore/<name> --gpus "$GPUS" \
     --mount "$(pwd)/runs/<tag>/models/<name>:/work" \
     --log runs/<tag>/models/<name>/run.log \
     --inner-cmd "<下载命令>"
   ```
6. 失败转阶段 D;成功 `progress.py set ... --stage B`。

## 阶段 C:验证推理

7. 有官方示例代码用示例;没有则 Claude 自建最小推理脚本,放进 `models/<name>/`。
8. 跑推理验证(对 dataset 产出 predictions.jsonl 到 /work/predictions.jsonl):
   ```bash
   uv run scripts/run_inference.py --tag autoexplore/<name> --gpus "$GPUS" \
     --run-dir runs/<tag>/models/<name> \
     --log runs/<tag>/models/<name>/run.log \
     --infer-cmd "<推理命令,输出写到 /work/predictions.jsonl>"
   ```
9. 成功 → `progress.py set ... --stage ready`,循环结束。失败 → 阶段 D。

## 阶段 D:有界重试

10. `retry_count += 1` 写进 progress.json。
11. 读 `tail -n 50 run.log` 的错误:
    - 简单可修(缺依赖/路径/CUDA 版本):改 Dockerfile 或命令,回阶段 A。
    - 根本性失败(架构不兼容/资源不够):放弃。
12. `retry_count >= 3` → 放弃,`progress.py set ... --stage crash`,在 results.tsv 记 crash:
    ```bash
    uv run scripts/progress.py result --run-dir runs/<tag> --model <name> \
      --primary-metric 0.0 --memory-gb 0.0 --status crash --description "<原因>"
    ```

## 成功后:推理 + 评测(步骤 8–9)

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
