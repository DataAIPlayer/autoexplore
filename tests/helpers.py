"""测试用纯函数(正常 import,勿放进 conftest 以免 import 坏味道)。"""
import json
from pathlib import Path


def write_predictions(path: Path, id_scores: dict):
    """按 {image_id: score} 写 predictions.jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sid, sc in id_scores.items():
            f.write(json.dumps({"image_id": sid, "score": sc, "layers": []}) + "\n")
