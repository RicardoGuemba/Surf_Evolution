"""Associa surfista e prancha (IoU ou proximidade de centros)."""

from __future__ import annotations

import math
from typing import Any

BBox = tuple[float, float, float, float]


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def center_distance(a: BBox, b: BBox) -> float:
    ca, cb = bbox_center(a), bbox_center(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def pair_person_surfboard(
    persons: list[dict[str, Any]],
    surfboards: list[dict[str, Any]],
    frame_diag: float,
    min_iou: float = 0.02,
    max_dist_frac: float = 0.35,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """
    Por pessoa (conf decrescente), melhor prancha livre.

    Maximiza IoU; se IoU < min_iou, usa distância entre centros até
    max_dist_frac * diagonal do frame.
    """
    if not persons:
        return []

    def iou(b1, b2) -> float:
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    sorted_p = sorted(persons, key=lambda d: d.get("conf", 0.0), reverse=True)
    used_board_idx: set[int] = set()
    max_dist = max(frame_diag * max_dist_frac, 1.0)
    out: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    for p in sorted_p:
        pb = p["bbox"]
        best_j: int | None = None
        best_score = -1.0
        for j, s in enumerate(surfboards):
            if j in used_board_idx:
                continue
            sb = s["bbox"]
            iv = iou(pb, sb)
            if iv >= min_iou:
                score = iv + p.get("conf", 0.0) * 1e-6
            else:
                dist = center_distance(pb, sb)
                if dist > max_dist:
                    continue
                score = (1.0 - dist / max_dist) * 0.5
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            used_board_idx.add(best_j)
            out.append((p, surfboards[best_j]))
        else:
            out.append((p, None))

    return out
