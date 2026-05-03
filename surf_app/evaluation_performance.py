"""
Avaliação de performance em tempo real (módulo separado da pontuação surf).

Parâmetros geométricos e temporais vêm de `EvaluationThresholds`
(`rules/evaluation_performance.json` + UI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from surf_app.evaluation_thresholds import EvaluationThresholds
from surf_app.poc_maneuvers import (
    LS,
    RS,
    LH,
    RH,
    LK,
    RK,
    LA,
    RA,
    _body_bbox_aspect,
    _k,
    _knee_angles,
    _mid_hip,
    _ok,
    _torso_angle_from_horizontal_deg,
)


def _body_axis_x(kp: Any, min_c: float) -> float | None:
    """Abcissa média do tronco (linha vertical imaginária pelo centro do corpo)."""
    xs: list[float] = []
    for i in (LS, RS, LH, RH):
        if _ok(_k(kp, i)[2], min_c):
            xs.append(_k(kp, i)[0])
    if len(xs) < 2:
        return None
    return sum(xs) / len(xs)


def _torso_length_px(kp: Any, min_c: float) -> float | None:
    """Comprimento aproximado do tronco (ombros–quadril) em pixels."""
    if not all(_ok(_k(kp, i)[2], min_c) for i in (LS, RS, LH, RH)):
        return None
    sx = (_k(kp, LS)[0] + _k(kp, RS)[0]) * 0.5
    sy = (_k(kp, LS)[1] + _k(kp, RS)[1]) * 0.5
    hx = (_k(kp, LH)[0] + _k(kp, RH)[0]) * 0.5
    hy = (_k(kp, LH)[1] + _k(kp, RH)[1]) * 0.5
    d = math.hypot(sx - hx, sy - hy)
    return d if d > 1e-3 else None


def _lateral_alignment_deviation_norm(kp: Any, min_c: float) -> float | None:
    """
    Desvio médio normalizado: |x_i - eixo_vertical| / comprimento_tronco.
    Quanto menor, melhor alinhamento à linha vertical do corpo.
    """
    axis = _body_axis_x(kp, min_c)
    tl = _torso_length_px(kp, min_c)
    if axis is None or tl is None:
        return None
    pts = (LS, RS, LK, RK, LA, RA)
    devs: list[float] = []
    for i in pts:
        if not _ok(_k(kp, i)[2], min_c):
            continue
        devs.append(abs(_k(kp, i)[0] - axis) / tl)
    if len(devs) < 2:
        return None
    return sum(devs) / len(devs)


def _alignment_quality_0_100(dev_norm: float | None, decay_k: float) -> float | None:
    """Converte desvio normalizado em score 0–100 (maior = melhor alinhamento)."""
    if dev_norm is None:
        return None
    k = max(1e-6, float(decay_k))
    s = 100.0 * math.exp(-k * float(dev_norm))
    return max(0.0, min(100.0, s))


@dataclass
class PerformanceEvaluationRuntime:
    """Estado e métricas em tempo real para o módulo de avaliação."""

    min_conf: float = 0.35
    thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)

    # Pop-up FSM
    _pop_prone_t0: float | None = None
    _pop_last_prone: float | None = None
    _pop_trans_end: float | None = None
    _last_transition_s: float | None = None
    _last_alignment_during_transition: float | None = None
    _align_samples: list[float] = field(default_factory=list)

    # In-pipeline instantâneo
    _knee_flex_score: float | None = None
    _hip_low_score: float | None = None
    _pipeline_combo: float | None = None

    popup_phase: str = "—"
    alignment_dev_norm: float | None = None
    alignment_score: float | None = None

    def reset(self) -> None:
        """Reinicia FSM e métricas instantâneas (não altera `thresholds`)."""
        self._pop_prone_t0 = None
        self._pop_last_prone = None
        self._pop_trans_end = None
        self._last_transition_s = None
        self._last_alignment_during_transition = None
        self._align_samples.clear()
        self._knee_flex_score = None
        self._hip_low_score = None
        self._pipeline_combo = None
        self.popup_phase = "—"
        self.alignment_dev_norm = None
        self.alignment_score = None

    def step(self, kp: Any, frame_wh: tuple[int, int], now: float) -> None:
        _fw, fh = frame_wh
        t = self.thresholds
        m = max(float(self.min_conf), float(t.evaluation_min_keypoint_confidence))

        prone = standing = False
        torso_h = _torso_angle_from_horizontal_deg(kp, m)
        asp = _body_bbox_aspect(kp, m)
        if torso_h is not None:
            prone = prone or torso_h < t.prone_torso_max_deg
            standing = standing or torso_h > t.standing_torso_min_deg
        if asp is not None:
            prone = prone or asp > t.prone_body_aspect_min
            standing = standing or asp < t.standing_body_aspect_max

        dev_now = _lateral_alignment_deviation_norm(kp, m)
        self.alignment_dev_norm = dev_now
        self.alignment_score = _alignment_quality_0_100(dev_now, t.alignment_score_decay_k)

        if prone:
            self.popup_phase = "prono"
            if self._pop_prone_t0 is None:
                self._pop_prone_t0 = now
            self._pop_last_prone = now
            if dev_now is not None:
                self._align_samples.append(dev_now)
        elif standing and self._pop_prone_t0 is not None and self._pop_last_prone is not None:
            prone_dur = self._pop_last_prone - self._pop_prone_t0
            total = now - self._pop_prone_t0
            if dev_now is not None:
                self._align_samples.append(dev_now)
            if prone_dur >= t.popup_prone_min_duration_s and total <= t.popup_max_transition_s:
                self._last_transition_s = float(total)
                if self._align_samples:
                    self._last_alignment_during_transition = sum(self._align_samples) / len(
                        self._align_samples
                    )
                self._pop_trans_end = now
            self._pop_prone_t0 = None
            self._pop_last_prone = None
            self._align_samples.clear()
            self.popup_phase = "em_pe"
        else:
            if self._pop_prone_t0 is not None:
                self.popup_phase = "transicao"
                if dev_now is not None:
                    self._align_samples.append(dev_now)
            if self._pop_prone_t0 is not None and (now - self._pop_prone_t0) > t.popup_fsm_timeout_s:
                self._pop_prone_t0 = None
                self._pop_last_prone = None
                self._align_samples.clear()
                self.popup_phase = "aguardando"
            elif self._pop_prone_t0 is None:
                self.popup_phase = "aguardando"

        lk, rk = _knee_angles(kp, m)
        hip = _mid_hip(kp, m)
        if lk is not None and rk is not None:
            flex_deg = max(0.0, 180.0 - float(lk)) + max(0.0, 180.0 - float(rk))
            div = max(40.0, float(t.pipeline_knee_flex_divisor_deg))
            self._knee_flex_score = min(100.0, flex_deg / div * 100.0)
        else:
            self._knee_flex_score = None

        if hip is not None and fh > 0:
            coef = max(50.0, float(t.pipeline_hip_vertical_coef))
            self._hip_low_score = min(100.0, max(0.0, (float(hip[1]) / float(fh)) * coef))
        else:
            self._hip_low_score = None

        wk = float(t.pipeline_weight_knee)
        wh = float(t.pipeline_weight_hip)
        if self._knee_flex_score is not None and self._hip_low_score is not None:
            self._pipeline_combo = wk * self._knee_flex_score + wh * self._hip_low_score
        elif self._knee_flex_score is not None:
            self._pipeline_combo = self._knee_flex_score
        elif self._hip_low_score is not None:
            self._pipeline_combo = self._hip_low_score
        else:
            self._pipeline_combo = None


def format_avaliacao_markdown(rt: PerformanceEvaluationRuntime) -> str:
    """Texto Markdown para o painel do módulo de avaliação."""
    th = rt.thresholds
    lines: list[str] = [
        "#### Pop-up",
        "",
        f"- **Fase:** `{rt.popup_phase}`",
    ]
    if rt._last_transition_s is not None:
        lines.append(f"- **Última transição deitado→em pé:** {rt._last_transition_s:.2f} s")
    else:
        lines.append("- **Última transição deitado→em pé:** —")
    if rt._last_alignment_during_transition is not None:
        lines.append(
            f"- **Alinhamento (desvio médio normalizado na transição):** "
            f"{rt._last_alignment_during_transition:.3f} _(menor = mais fluido/alinhado)_"
        )
    else:
        lines.append("- **Alinhamento na última transição:** —")
    if rt.alignment_dev_norm is not None:
        lines.append(
            f"- **Alinhamento instantâneo:** desvio {rt.alignment_dev_norm:.3f} · "
            f"score {rt.alignment_score:.0f}/100"
        )
    else:
        lines.append("- **Alinhamento instantâneo:** —")
    lines.extend(
        [
            "",
            "#### In-pipeline",
            "",
        ]
    )
    if rt._knee_flex_score is not None:
        lines.append(f"- **Agachamento (joelhos):** {rt._knee_flex_score:.0f}/100 _(maior = mais flexionado)_")
    else:
        lines.append("- **Agachamento (joelhos):** —")
    if rt._hip_low_score is not None:
        lines.append(f"- **Quadril próximo do chão:** {rt._hip_low_score:.0f}/100 _(maior = mais baixo no quadro)_")
    else:
        lines.append("- **Quadril próximo do chão:** —")
    if rt._pipeline_combo is not None:
        lines.append(f"- **Combinado in-pipeline:** {rt._pipeline_combo:.0f}/100")
    else:
        lines.append("- **Combinado in-pipeline:** —")
    lines.extend(
        [
            "",
            "#### Parâmetros ativos (resumo)",
            "",
            f"- Torso prono/em pé: `<{th.prone_torso_max_deg:.0f}°` / `>{th.standing_torso_min_deg:.0f}°`",
            f"- Aspeto corpo prono/em pé: `>{th.prone_body_aspect_min:.2f}` / `<{th.standing_body_aspect_max:.2f}`",
            f"- Tempos pop-up: mín. prono **{th.popup_prone_min_duration_s:.2f}s**, transição máx. **{th.popup_max_transition_s:.2f}s**, timeout FSM **{th.popup_fsm_timeout_s:.2f}s**",
            f"- Conf. mín. avaliação: **{th.evaluation_min_keypoint_confidence:.2f}**",
        ]
    )
    return "\n".join(lines)
