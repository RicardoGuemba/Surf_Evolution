"""Motor mínimo de regras JSON para POC (manobras + pontuação)."""

from __future__ import annotations

import json
import time
from typing import Any


class RulesParseError(ValueError):
    pass


def parse_rules_json(text: str) -> dict[str, Any]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise RulesParseError(str(e)) from e
    if not isinstance(doc, dict):
        raise RulesParseError("raiz deve ser um objeto JSON")
    return doc


def _cmp(op: str, left: float, right: float) -> bool:
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == "==":
        return abs(left - right) < 1e-6
    raise RulesParseError(f"operador não suportado: {op}")


class RulesRuntime:
    """Avalia manobras com condições simultâneas e cooldown."""

    def __init__(self, doc: dict[str, Any]):
        self.doc = doc
        g = doc.get("global") or {}
        self.min_kpt_conf = float(g.get("min_keypoint_confidence", 0.35))
        self.maneuvers: list[dict[str, Any]] = list(doc.get("maneuvers") or [])
        self.fps_assumption = float(doc.get("sampling_fps_assumption") or 30.0)
        self._cooldown_until: dict[str, float] = {}
        self._hold_since: dict[str, float] = {}

    def global_min_kpt_confidence(self) -> float:
        return self.min_kpt_conf

    def _conditions_met(self, features: dict[str, float], conds: list[dict[str, Any]]) -> bool:
        if not conds:
            return False
        for c in conds:
            name = c.get("feature")
            if name not in features:
                return False
            op = c.get("op", ">=")
            val = float(c.get("value", 0))
            if not _cmp(op, float(features[name]), val):
                return False
        return True

    def _required_hold_s(self, conds: list[dict[str, Any]]) -> float:
        if not conds:
            return 0.0
        ms = max(float(c.get("duration_ms", 0) or 0) for c in conds)
        return ms / 1000.0

    def _score(self, m: dict[str, Any], features: dict[str, float]) -> float:
        sc = m.get("score") or {}
        base = float(sc.get("base", 0))
        total = base
        for mult in sc.get("multipliers") or []:
            feat = mult.get("feature")
            k = float(mult.get("k", 0))
            if feat in features:
                total += k * float(features[feat])
        return max(0.0, total)

    def tick(self, features: dict[str, float], now: float | None = None) -> list[dict[str, Any]]:
        """Retorna eventos disparados neste instante."""
        now = now if now is not None else time.monotonic()
        fired: list[dict[str, Any]] = []

        avg_c = float(features.get("avg_kpt_conf", 0))
        if avg_c < self.min_kpt_conf:
            self._hold_since.clear()
            return fired

        for m in self.maneuvers:
            mid = str(m.get("id", ""))
            if not mid:
                continue
            if now < self._cooldown_until.get(mid, 0.0):
                continue
            conds = list(m.get("conditions") or [])
            if not self._conditions_met(features, conds):
                self._hold_since.pop(mid, None)
                continue

            need_s = self._required_hold_s(conds)
            start = self._hold_since.get(mid)
            if start is None:
                self._hold_since[mid] = now
                start = now
            held = now - start
            if held >= need_s:
                pts = self._score(m, features)
                fired.append(
                    {
                        "maneuver_id": mid,
                        "label": m.get("label", mid),
                        "score": pts,
                        "features_snapshot": {k: float(v) for k, v in features.items()},
                    }
                )
                cd_s = float(m.get("cooldown_ms", 1000) or 1000) / 1000.0
                self._cooldown_until[mid] = now + cd_s
                self._hold_since.pop(mid, None)

        return fired

    def reset_hold(self) -> None:
        self._hold_since.clear()
