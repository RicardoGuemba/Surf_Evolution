"""
Manobras POC com estado temporal (keypoints COCO).

Referência de onda: vem da ESQUERDA da imagem (x menor).
- Front-side (de frente para a onda): nariz à esquerda do quadril (olha para a esquerda).
- Back-side (de costas): nariz à direita do quadril (olha para a direita).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


NOSE = 0
LS, RS = 5, 6
LH, RH = 11, 12
LK, RK = 13, 14
LA, RA = 15, 16


def _k(kp: Any, i: int) -> tuple[float, float, float]:
    return float(kp[i][0]), float(kp[i][1]), float(kp[i][2])


def _ok(c: float, m: float) -> bool:
    return c >= m and not math.isnan(c)


def _angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Ângulo interno em B (graus)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(ba[0], ba[1])
    n2 = math.hypot(bc[0], bc[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    d = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (n1 * n2)))
    return math.degrees(math.acos(d))


def _torso_angle_from_horizontal_deg(kp: Any, min_c: float) -> float | None:
    if not all(_ok(_k(kp, i)[2], min_c) for i in (LS, RS, LH, RH)):
        return None
    sx = (_k(kp, LS)[0] + _k(kp, RS)[0]) * 0.5
    sy = (_k(kp, LS)[1] + _k(kp, RS)[1]) * 0.5
    hx = (_k(kp, LH)[0] + _k(kp, RH)[0]) * 0.5
    hy = (_k(kp, LH)[1] + _k(kp, RH)[1]) * 0.5
    vx, vy = sx - hx, sy - hy
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    vx, vy = vx / n, vy / n
    return abs(math.degrees(math.atan2(vy, vx)))


def _body_bbox_aspect(kp: Any, min_c: float) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(17):
        x, y, c = _k(kp, i)
        if _ok(c, min_c):
            xs.append(x)
            ys.append(y)
    if len(xs) < 6:
        return None
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h < 1e-6:
        return None
    return w / h


def _mid_hip(kp: Any, min_c: float) -> tuple[float, float] | None:
    if not _ok(_k(kp, LH)[2], min_c) or not _ok(_k(kp, RH)[2], min_c):
        return None
    return (_k(kp, LH)[0] + _k(kp, RH)[0]) * 0.5, (_k(kp, LH)[1] + _k(kp, RH)[1]) * 0.5


def _knee_angles(kp: Any, min_c: float) -> tuple[float | None, float | None]:
    """Ângulos internos nos joelhos (180 = esticado)."""
    lknee = rknee = None
    if all(_ok(_k(kp, i)[2], min_c) for i in (LH, LK, LA)):
        lknee = _angle_deg((_k(kp, LH)[0], _k(kp, LH)[1]), (_k(kp, LK)[0], _k(kp, LK)[1]), (_k(kp, LA)[0], _k(kp, LA)[1]))
    if all(_ok(_k(kp, i)[2], min_c) for i in (RH, RK, RA)):
        rknee = _angle_deg((_k(kp, RH)[0], _k(kp, RH)[1]), (_k(kp, RK)[0], _k(kp, RK)[1]), (_k(kp, RA)[0], _k(kp, RA)[1]))
    return lknee, rknee


@dataclass
class SurfPocManeuverEngine:
    """Detecta as 4 manobras POC e expõe dica de estado para a UI."""

    min_conf: float = 0.35
    wave_from_left: bool = True  # onda na esquerda do frame

    ui_hint_lines: list[str] = field(default_factory=list)

    _pop_prone_t0: float | None = None
    _pop_last_prone: float | None = None
    _pop_cd_until: float = 0.0

    _hip_trace: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=120))
    _cut_cd_until: float = 0.0

    _pipe_back_t0: float | None = None
    _pipe_front_t0: float | None = None
    _pipe_back_fired: bool = False
    _pipe_front_fired: bool = False
    _pipe_back_cd: float = 0.0
    _pipe_front_cd: float = 0.0

    def reset(self) -> None:
        self.ui_hint_lines = []
        self._pop_prone_t0 = None
        self._pop_last_prone = None
        self._pop_cd_until = 0.0
        self._hip_trace.clear()
        self._cut_cd_until = 0.0
        self._pipe_back_t0 = None
        self._pipe_front_t0 = None
        self._pipe_back_fired = False
        self._pipe_front_fired = False
        self._pipe_back_cd = 0.0
        self._pipe_front_cd = 0.0

    def step(
        self,
        kp: Any,
        frame_wh: tuple[int, int],
        now: float,
    ) -> list[dict[str, Any]]:
        self.ui_hint_lines = []
        out: list[dict[str, Any]] = []
        fw, fh = frame_wh
        m = self.min_conf

        asp = _body_bbox_aspect(kp, m)
        torso_h = _torso_angle_from_horizontal_deg(kp, m)
        prone = False
        standing = False
        if torso_h is not None:
            prone = prone or torso_h < 48
            standing = standing or torso_h > 56
        if asp is not None:
            prone = prone or asp > 1.15
            standing = standing or asp < 0.90

        hip = _mid_hip(kp, m)
        if hip is not None:
            self._hip_trace.append((now, hip[0]))

        # --- Pop-up (2 pts): deitado ≥0,12 s → em pé; transição em < 3 s desde o início do deitado ---
        if now >= self._pop_cd_until:
            if prone:
                if self._pop_prone_t0 is None:
                    self._pop_prone_t0 = now
                self._pop_last_prone = now
            elif standing and self._pop_prone_t0 is not None and self._pop_last_prone is not None:
                prone_dur = self._pop_last_prone - self._pop_prone_t0
                total = now - self._pop_prone_t0
                if prone_dur >= 0.12 and total <= 3.0:
                    out.append(
                        {
                            "maneuver_id": "popup",
                            "label": "Movimento pop-up",
                            "score": 2.0,
                            "source": "poc",
                        }
                    )
                    self._pop_cd_until = now + 4.0
                self._pop_prone_t0 = None
                self._pop_last_prone = None

            if self._pop_prone_t0 is not None and (now - self._pop_prone_t0) > 3.2:
                self._pop_prone_t0 = None
                self._pop_last_prone = None

        if prone:
            self.ui_hint_lines.append("Pop-up: deitado detectado — levante em até 3 s para +2 pts")

        # --- Cut-back em S (4 pts): trajetória do quadril em S ---
        if now >= self._cut_cd_until and hip is not None and len(self._hip_trace) >= 25:
            xs = [x for t, x in self._hip_trace if now - t <= 3.2]
            if len(xs) >= 25:
                # suavização
                k = 5
                sm = [sum(xs[i : i + k]) / k for i in range(len(xs) - k + 1)]
                if len(sm) >= 12:
                    dx = [sm[i + 1] - sm[i] for i in range(len(sm) - 1)]
                    signs = [1 if d > 0.4 else (-1 if d < -0.4 else 0) for d in dx]
                    nz = [(i, s) for i, s in enumerate(signs) if s != 0]
                    if len(nz) >= 4:
                        seq = [s for _, s in nz]
                        changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
                        amp = max(sm) - min(sm)
                        if changes >= 3 and amp > 0.055 * fw:
                            out.append(
                                {
                                    "maneuver_id": "cut_back_s",
                                    "label": "Manobra cut-back (trajetória em S)",
                                    "score": 4.0,
                                    "source": "poc",
                                }
                            )
                            self._cut_cd_until = now + 5.0
                            self._hip_trace.clear()
        if hip is not None and 15 < len(self._hip_trace) < 90 and now >= self._cut_cd_until:
            self.ui_hint_lines.append("Cut-back: trace S com o quadril (~2–3 s)")

        # --- Pipeline (8 pts): joelhos ~45° de flexão (≈135° interno), segurar ≥1 s e ≤4 s ---
        nose_c = _k(kp, NOSE)[2]
        hip_pt = _mid_hip(kp, m)
        lk, rk = _knee_angles(kp, m)

        def knees_ok() -> bool:
            if lk is None or rk is None:
                return False
            target = 135.0
            tol = 16.0
            return abs(lk - target) <= tol and abs(rk - target) <= tol

        facing_wave = False
        back_to_wave = False
        margin = 14.0
        if _ok(nose_c, m) and hip_pt is not None:
            nx = _k(kp, NOSE)[0]
            if self.wave_from_left:
                facing_wave = nx + margin < hip_pt[0]
                back_to_wave = nx > hip_pt[0] + margin
            else:
                facing_wave = nx > hip_pt[0] + margin
                back_to_wave = nx + margin < hip_pt[0]

        if now >= self._pipe_back_cd:
            if back_to_wave and knees_ok():
                if self._pipe_back_t0 is None:
                    self._pipe_back_t0 = now
                    self._pipe_back_fired = False
                elapsed = now - self._pipe_back_t0
                if elapsed > 4.25:
                    self._pipe_back_t0 = None
                    self._pipe_back_fired = False
                elif elapsed >= 1.0 and not self._pipe_back_fired:
                    out.append(
                        {
                            "maneuver_id": "pipeline_backside",
                            "label": "In-pipeline back-side",
                            "score": 8.0,
                            "source": "poc",
                        }
                    )
                    self._pipe_back_fired = True
                    self._pipe_back_cd = now + 5.0
                    self._pipe_back_t0 = None
            else:
                self._pipe_back_t0 = None
                self._pipe_back_fired = False

        if back_to_wave and knees_ok():
            self.ui_hint_lines.append(
                "Back-side (costas à onda): flexão ~45° nos joelhos — segure 1–4 s (+8)"
            )

        if now >= self._pipe_front_cd:
            if facing_wave and knees_ok():
                if self._pipe_front_t0 is None:
                    self._pipe_front_t0 = now
                    self._pipe_front_fired = False
                elapsed = now - self._pipe_front_t0
                if elapsed > 4.25:
                    self._pipe_front_t0 = None
                    self._pipe_front_fired = False
                elif elapsed >= 1.0 and not self._pipe_front_fired:
                    out.append(
                        {
                            "maneuver_id": "pipeline_frontside",
                            "label": "In-pipeline front-side",
                            "score": 8.0,
                            "source": "poc",
                        }
                    )
                    self._pipe_front_fired = True
                    self._pipe_front_cd = now + 5.0
                    self._pipe_front_t0 = None
            else:
                self._pipe_front_t0 = None
                self._pipe_front_fired = False

        if facing_wave and knees_ok():
            self.ui_hint_lines.append(
                "Front-side (frente à onda): flexão ~45° nos joelhos — segure 1–4 s (+8)"
            )

        return out
