import json
import sys
import subprocess
from pathlib import Path

from scripts import directions_schema as dsx


def _valid():
    return [
        {"slot": "a", "title": "Method A", "source_urls": ["http://arxiv.org/abs/1"],
         "idea": "do x", "tier": "config", "needs_training": False},
        {"slot": "b", "title": "Method B", "source_urls": [],
         "idea": "do y", "tier": "train", "needs_training": True},
    ]


def test_validate_accepts_valid():
    assert dsx.validate(_valid()) == []


def test_validate_rejects_bad_tier():
    d = _valid()
    d[0]["tier"] = "magic"
    errs = dsx.validate(d)
    assert any("tier" in e for e in errs)


def test_validate_rejects_missing_field():
    d = _valid()
    del d[1]["idea"]
    errs = dsx.validate(d)
    assert any("idea" in e for e in errs)


def test_validate_rejects_duplicate_slots():
    d = _valid()
    d[1]["slot"] = "a"
    errs = dsx.validate(d)
    assert any("duplicate" in e for e in errs)


def test_validate_rejects_empty():
    assert dsx.validate([]) != []


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid()))
    r = subprocess.run([sys.executable, "scripts/directions_schema.py",
                        "--file", str(good)], capture_output=True, text=True)
    assert r.returncode == 0
    bad = tmp_path / "bad.json"
    d = _valid(); d[0]["tier"] = "magic"
    bad.write_text(json.dumps(d))
    r2 = subprocess.run([sys.executable, "scripts/directions_schema.py",
                         "--file", str(bad)], capture_output=True, text=True)
    assert r2.returncode == 1
    assert "tier" in r2.stdout + r2.stderr


def test_phase3_tiers_accepted():
    dirs = [{"slot": "a", "title": "FP8", "source_urls": ["u1"], "idea": "quant",
             "tier": "quantization", "needs_training": False}]
    assert dsx.validate(dirs, dsx.PHASE3_TIERS) == []


def test_phase3_tier_rejected_under_phase2_set():
    dirs = [{"slot": "a", "title": "FP8", "source_urls": ["u1"], "idea": "quant",
             "tier": "quantization", "needs_training": False}]
    errs = dsx.validate(dirs, dsx.PHASE2_TIERS)
    assert any("tier must be one of" in e for e in errs)
