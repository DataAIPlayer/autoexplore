"""短板诊断 (只读,不改冻结 evaluate.py)。把 evaluate.py 当黑盒,
在 dataset 子集视图上反复调用,定位主干短板。绝不重实现指标。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_evaluate(evaluate_py: Path, predictions: Path, dataset_view: Path,
                  out_json: Path) -> dict:
    """黑盒调用冻结 evaluate.py。失败或缺 primary_metric 时报清晰错误(带视图名)。"""
    proc = subprocess.run([sys.executable, str(evaluate_py),
                           "--predictions", str(predictions),
                           "--dataset", str(dataset_view),
                           "--out", str(out_json)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate.py 在视图 {dataset_view.name} 失败:\n{(proc.stderr or '').strip()}")
    r = json.loads(out_json.read_text())
    if "primary_metric" not in r:
        raise RuntimeError(f"evaluate.py 输出缺 primary_metric (视图 {dataset_view.name})")
    return r


def _detect_id_field(samples: list[dict], dataset_dir: Path,
                     override: str | None = None) -> str:
    """探测样本标识字段:值唯一、且每个值对应 dataset_dir 下一个子目录。
    优先 image_id(本项目惯例);override 显式指定时直接用。
    通用——不把字段名绑死成 image_id(不同任务可能用 id/uid/utt_id 等)。"""
    if override:
        return override
    if not samples:
        return "image_id"
    common = set(samples[0].keys())
    for s in samples[1:]:
        common &= set(s.keys())

    def ok(k: str) -> bool:
        vals = [s.get(k) for s in samples]
        if any(not isinstance(v, (str, int)) for v in vals):
            return False
        strs = [str(v) for v in vals]
        if len(set(strs)) != len(strs):  # 必须唯一
            return False
        return all((dataset_dir / v).is_dir() for v in strs)  # 且对应样本目录

    if "image_id" in common and ok("image_id"):
        return "image_id"
    for k in sorted(common):
        if ok(k):
            return k
    return "image_id"  # 兜底


def _make_view(dataset_dir: Path, sample_ids: list[str], view_dir: Path,
               id_field: str = "image_id") -> None:
    """建子集视图:子集化 metadata.json + symlink 样本目录。原 dataset 不动。"""
    view_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    keep = set(sample_ids)
    sub = {k: v for k, v in meta.items() if k != "samples"}
    sub["samples"] = [s for s in meta["samples"] if s[id_field] in keep]
    (view_dir / "metadata.json").write_text(json.dumps(sub, ensure_ascii=False))
    for sid in sample_ids:
        link = view_dir / sid
        if not link.is_symlink():  # is_symlink 对断链也为真,避免 FileExistsError
            os.symlink((dataset_dir / sid).resolve(), link)


def _detect_group_fields(samples: list[dict], id_field: str = "image_id",
                         max_card_ratio: float = 0.5) -> list[str]:
    """自动探测可分组的分类字段:所有样本都有、值为 str/int/bool、2≤基数≤0.5*n。
    排除标识字段 id_field 本身。"""
    n = len(samples)
    if n == 0:
        return []
    keys = set().union(*[set(s.keys()) for s in samples]) - {id_field}
    fields = []
    for k in sorted(keys):
        vals = [s.get(k) for s in samples if isinstance(s.get(k), (str, int, bool))]
        if len(vals) != n:
            continue
        distinct = set(vals)
        if 2 <= len(distinct) <= max(2, int(n * max_card_ratio)):
            fields.append(k)
    return fields


def diagnose(predictions: Path, dataset_dir: Path, evaluate_py: Path,
             worst_k: int = 10, work_dir: Path | None = None,
             id_field: str | None = None) -> dict:
    """对当前主干的 predictions 做短板分解。返回值只含解析后的数值,不含临时文件路径,
    因此 work_dir 缺省时用临时目录跑完即清理(不泄漏 /tmp)。
    id_field=None 时自动探测样本标识字段(通用,不绑死 image_id)。"""
    if work_dir is not None:
        tmp = Path(work_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        return _diagnose_into(tmp, predictions, dataset_dir, evaluate_py, worst_k, id_field)
    with tempfile.TemporaryDirectory(prefix="diag_") as td:
        return _diagnose_into(Path(td), predictions, dataset_dir, evaluate_py,
                              worst_k, id_field)


def _diagnose_into(tmp: Path, predictions: Path, dataset_dir: Path,
                   evaluate_py: Path, worst_k: int,
                   id_field: str | None = None) -> dict:
    # 视图目录/输出文件一律用整数序号命名,避免标识/分组值里的 '/' 空格
    # 造成嵌套目录或命名冲突;真实 id / 分组值只作为返回数据的 key 保留。
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    samples = meta["samples"]
    idf = _detect_id_field(samples, dataset_dir, id_field)
    ids = [s[idf] for s in samples]

    full = _run_evaluate(evaluate_py, predictions, dataset_dir, tmp / "full.json")

    per_sample = []
    for i, sid in enumerate(ids):
        v = tmp / f"view_s{i}"
        _make_view(dataset_dir, [sid], v, idf)
        r = _run_evaluate(evaluate_py, predictions, v, tmp / f"s{i}.json")
        per_sample.append({"id": sid, "primary": r["primary_metric"]})
    per_sample.sort(key=lambda x: x["primary"])
    worst = per_sample[:worst_k]

    groups: dict[str, dict] = {}
    for fi, field in enumerate(_detect_group_fields(samples, idf)):
        by_val: dict[str, list[str]] = {}
        for s in samples:
            by_val.setdefault(str(s[field]), []).append(s[idf])
        gmap = {}
        for gi, (val, sids) in enumerate(by_val.items()):
            v = tmp / f"view_g{fi}_{gi}"
            _make_view(dataset_dir, sids, v, idf)
            r = _run_evaluate(evaluate_py, predictions, v, tmp / f"g{fi}_{gi}.json")
            gmap[val] = r["primary_metric"]
        groups[field] = gmap

    return {"full": full, "per_sample": per_sample, "worst_k": worst,
            "groups": groups, "secondary_summary": full.get("metrics", {})}


def _to_md(diag: dict) -> str:
    lines = ["# 短板诊断", "",
             f"- primary_metric (全集): {diag['full']['primary_metric']:.6f}",
             "- 副指标: " + json.dumps(diag["secondary_summary"], ensure_ascii=False),
             "", "## 最差样本 (worst-K)"]
    for w in diag["worst_k"]:
        lines.append(f"- {w['id']}: {w['primary']:.6f}")
    if diag["groups"]:
        lines.append("\n## 分组得分")
        for field, gmap in diag["groups"].items():
            lines.append(f"### {field}")
            for val, sc in sorted(gmap.items(), key=lambda kv: kv[1]):
                lines.append(f"- {val}: {sc:.6f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="短板诊断 (黑盒调用冻结 evaluate.py)")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--evaluate-py", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--worst-k", type=int, default=10)
    ap.add_argument("--id-field", default=None,
                    help="样本标识字段名;缺省自动探测(通用,不绑死 image_id)")
    args = ap.parse_args(argv)
    diag = diagnose(args.predictions, args.dataset, args.evaluate_py,
                    args.worst_k, id_field=args.id_field)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2))
    args.out_md.write_text(_to_md(diag))
    print(f"diagnosed: primary={diag['full']['primary_metric']:.6f}, "
          f"worst={len(diag['worst_k'])}, groups={list(diag['groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
