import time

from surf_app.rules_engine import RulesRuntime, parse_rules_json


def test_parse_rules():
    doc = parse_rules_json(
        '{"version":"1","sampling_fps_assumption":30,"global":{"min_keypoint_confidence":0.1},'
        '"maneuvers":[{"id":"t","label":"T","cooldown_ms":100,"score":{"base":10},"conditions":[]}]}'
    )
    assert doc["version"] == "1"


def test_no_conditions_never_fires():
    r = RulesRuntime(
        {
            "global": {"min_keypoint_confidence": 0.1},
            "maneuvers": [
                {
                    "id": "bad",
                    "label": "Bad",
                    "cooldown_ms": 0,
                    "score": {"base": 99},
                    "conditions": [],
                }
            ],
        }
    )
    assert not r.tick({"avg_kpt_conf": 1.0})


def test_fires_after_hold_and_respects_cooldown():
    r = RulesRuntime(
        {
            "global": {"min_keypoint_confidence": 0.1},
            "maneuvers": [
                {
                    "id": "lean",
                    "label": "Lean",
                    "cooldown_ms": 1000,
                    "score": {"base": 50, "multipliers": [{"feature": "turn_intensity", "k": 2}]},
                    "conditions": [
                        {"feature": "torso_lean_deg", "op": ">=", "value": 10, "duration_ms": 50},
                    ],
                }
            ],
        }
    )
    t0 = time.monotonic()
    assert r.tick({"avg_kpt_conf": 1.0, "torso_lean_deg": 5.0}, t0) == []
    assert r.tick({"avg_kpt_conf": 1.0, "torso_lean_deg": 20.0}, t0 + 0.01) == []
    fired = r.tick({"avg_kpt_conf": 1.0, "torso_lean_deg": 20.0}, t0 + 0.06)
    assert len(fired) == 1
    assert fired[0]["maneuver_id"] == "lean"
    assert r.tick({"avg_kpt_conf": 1.0, "torso_lean_deg": 20.0}, t0 + 0.07) == []
