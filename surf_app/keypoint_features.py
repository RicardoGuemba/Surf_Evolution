"""Features derivadas de keypoints COCO (17 pontos) para regras de manobra."""

from __future__ import annotations

import math
from typing import Any


def _conf_ok(c: float, min_c: float) -> bool:
    return c >= min_c and not math.isnan(c)


def features_from_pose_xy(
    kpts: Any,
    min_conf: float,
    prev: dict[str, float] | None,
    dt_s: float,
) -> dict[str, float]:
    """
    kpts: array (17, 3) — x, y, confidence (YOLO COCO).
    prev: estado anterior com chaves hip_line_angle_deg, timestamp opcional.
    dt_s: delta tempo desde frame anterior (>= 1e-6).
    """
    prev = prev or {}
    out: dict[str, float] = {
        "avg_kpt_conf": 0.0,
        "torso_lean_deg": 0.0,
        "hip_rotation_deg_s": 0.0,
        "turn_intensity": 0.0,
        "knee_flex_mean_deg": 0.0,
    }

    if kpts is None or len(kpts) < 17:
        return out

    kp = kpts
    confs = [float(kp[i][2]) for i in range(17)]
    out["avg_kpt_conf"] = sum(confs) / 17.0

    ls, rs = 5, 6
    lh, rh = 11, 12
    lk, rk = 13, 14

    need = [ls, rs, lh, rh]
    if not all(_conf_ok(float(kp[i][2]), min_conf) for i in need):
        return out

    sx = (float(kp[ls][0]) + float(kp[rs][0])) * 0.5
    sy = (float(kp[ls][1]) + float(kp[rs][1])) * 0.5
    hx = (float(kp[lh][0]) + float(kp[rh][0])) * 0.5
    hy = (float(kp[lh][1]) + float(kp[rh][1])) * 0.5
    vx, vy = sx - hx, sy - hy
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return out
    vx, vy = vx / norm, vy / norm
    # Inclinação do tronco em relação ao vertical (0, -1): 0 = ereto
    vert_x, vert_y = 0.0, -1.0
    dot = max(-1.0, min(1.0, vx * vert_x + vy * vert_y))
    lean = math.degrees(math.acos(dot))
    out["torso_lean_deg"] = lean
    out["turn_intensity"] = min(lean / 45.0, 2.0)

    # Linha do quadril no plano da imagem — taxa angular
    dhx = float(kp[rh][0]) - float(kp[lh][0])
    dhy = float(kp[rh][1]) - float(kp[lh][1])
    hip_ang = math.degrees(math.atan2(dhy, dhx))
    prev_ang = prev.get("hip_line_angle_deg")
    if prev_ang is not None and dt_s >= 1e-4:
        diff = hip_ang - prev_ang
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        out["hip_rotation_deg_s"] = abs(diff) / dt_s
    out["_hip_line_angle_deg"] = hip_ang

    if _conf_ok(float(kp[lk][2]), min_conf) and _conf_ok(float(kp[rk][2]), min_conf):
        def knee_angle(hx, hy, kx, ky, ax, ay):
            v1 = (hx - kx, hy - ky)
            v2 = (ax - kx, ay - ky)
            n1 = math.hypot(v1[0], v1[1])
            n2 = math.hypot(v2[0], v2[1])
            if n1 < 1e-6 or n2 < 1e-6:
                return 180.0
            d = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            return math.degrees(math.acos(d))

        la = 15
        ra = 16
        if _conf_ok(float(kp[la][2]), min_conf):
            fk = knee_angle(
                float(kp[lh][0]),
                float(kp[lh][1]),
                float(kp[lk][0]),
                float(kp[lk][1]),
                float(kp[la][0]),
                float(kp[la][1]),
            )
        else:
            fk = 180.0
        if _conf_ok(float(kp[ra][2]), min_conf):
            rknee = knee_angle(
                float(kp[rh][0]),
                float(kp[rh][1]),
                float(kp[rk][0]),
                float(kp[rk][1]),
                float(kp[ra][0]),
                float(kp[ra][1]),
            )
        else:
            rknee = 180.0
        out["knee_flex_mean_deg"] = (fk + rknee) * 0.5

    return out
