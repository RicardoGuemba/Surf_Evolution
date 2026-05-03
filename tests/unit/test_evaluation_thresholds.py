"""Validação e I/O de `EvaluationThresholds`."""

from __future__ import annotations

import json
from pathlib import Path

from surf_app.evaluation_thresholds import (
    EvaluationThresholds,
    default_thresholds_path,
    load_thresholds,
    save_thresholds,
    thresholds_from_dict,
    validate_thresholds,
)


def test_validate_thresholds_ok_defaults() -> None:
    assert validate_thresholds(EvaluationThresholds()) == []


def test_validate_thresholds_detects_conflicts() -> None:
    t = EvaluationThresholds(
        prone_torso_max_deg=50.0,
        standing_torso_min_deg=40.0,
    )
    errs = validate_thresholds(t)
    assert any("prone_torso_max_deg" in e for e in errs)


def test_thresholds_from_dict_ignores_unknown_keys() -> None:
    t = thresholds_from_dict({"version": "9", "unknown": 123, "prone_torso_max_deg": 12.0})
    assert t.version == "9"
    assert t.prone_torso_max_deg == 12.0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    t = EvaluationThresholds(
        prone_torso_max_deg=35.0,
        popup_prone_min_duration_s=0.5,
    )
    save_thresholds(t, project_root=str(root))
    p = default_thresholds_path(str(root))
    assert p.is_file()
    t2 = load_thresholds(project_root=str(root))
    assert t2.prone_torso_max_deg == 35.0
    assert t2.popup_prone_min_duration_s == 0.5
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["prone_torso_max_deg"] == 35.0
