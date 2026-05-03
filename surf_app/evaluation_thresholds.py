"""
Parâmetros tunáveis da avaliação de performance (pop-up, in-pipeline).

Persistência: `rules/evaluation_performance.json` (versionado no repositório).
Validação explícita antes de aplicar na sessão (fail-fast, mensagens claras).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_REL_PATH = "rules/evaluation_performance.json"


@dataclass
class EvaluationThresholds:
    """Limites temporais e geométricos — valores mais estritos = deteção mais exigente."""

    version: str = "1.0"
    # Pop-up: critérios de postura (graus / razão largura-altura do bbox de keypoints)
    prone_torso_max_deg: float = 40.0
    standing_torso_min_deg: float = 64.0
    prone_body_aspect_min: float = 1.28
    standing_body_aspect_max: float = 0.78
    # Pop-up: tempos (s)
    popup_prone_min_duration_s: float = 0.35
    popup_max_transition_s: float = 2.6
    popup_fsm_timeout_s: float = 3.1
    # Alinhamento: exp(-k * desvio_norm) → 100
    alignment_score_decay_k: float = 8.0
    # In-pipeline: normalização e pesos
    pipeline_knee_flex_divisor_deg: float = 120.0
    pipeline_hip_vertical_coef: float = 130.0
    pipeline_weight_knee: float = 0.55
    pipeline_weight_hip: float = 0.45
    # Confiança mínima dos keypoints só neste módulo (não altera rules_engine)
    evaluation_min_keypoint_confidence: float = 0.38


def default_thresholds_path(project_root: str | None = None) -> Path:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    return root / DEFAULT_REL_PATH


def _field_names() -> set[str]:
    return {f.name for f in fields(EvaluationThresholds)}


def thresholds_from_dict(data: dict[str, Any]) -> EvaluationThresholds:
    """Constrói a partir de dict JSON (ignora chaves desconhecidas)."""
    names = _field_names()
    kwargs = {k: data[k] for k in data if k in names}
    return EvaluationThresholds(**kwargs)


def validate_thresholds(t: EvaluationThresholds) -> list[str]:
    """Lista de erros humanos; vazia = OK."""
    err: list[str] = []
    if t.prone_torso_max_deg >= t.standing_torso_min_deg:
        err.append("prone_torso_max_deg deve ser menor que standing_torso_min_deg.")
    if t.prone_body_aspect_min <= 1.0:
        err.append("prone_body_aspect_min deve ser > 1.0 (corpo mais largo que alto).")
    if t.standing_body_aspect_max >= 1.2:
        err.append("standing_body_aspect_max deve ser < 1.2 (corpo mais alto que largo).")
    if t.popup_prone_min_duration_s <= 0 or t.popup_prone_min_duration_s > 2.0:
        err.append("popup_prone_min_duration_s deve estar em ]0, 2].")
    if t.popup_max_transition_s <= 0 or t.popup_max_transition_s > 8.0:
        err.append("popup_max_transition_s deve estar em ]0, 8].")
    if t.popup_fsm_timeout_s < t.popup_max_transition_s:
        err.append("popup_fsm_timeout_s deve ser >= popup_max_transition_s.")
    if t.alignment_score_decay_k <= 0 or t.alignment_score_decay_k > 30:
        err.append("alignment_score_decay_k deve estar em ]0, 30].")
    if t.pipeline_knee_flex_divisor_deg < 40 or t.pipeline_knee_flex_divisor_deg > 200:
        err.append("pipeline_knee_flex_divisor_deg sugerido em [40, 200].")
    if t.pipeline_hip_vertical_coef < 50 or t.pipeline_hip_vertical_coef > 200:
        err.append("pipeline_hip_vertical_coef sugerido em [50, 200].")
    wk, wh = t.pipeline_weight_knee, t.pipeline_weight_hip
    if wk < 0 or wh < 0 or abs(wk + wh - 1.0) > 0.02:
        err.append("pipeline_weight_knee + pipeline_weight_hip deve somar ~1.0.")
    if t.evaluation_min_keypoint_confidence < 0.1 or t.evaluation_min_keypoint_confidence > 0.95:
        err.append("evaluation_min_keypoint_confidence em [0.1, 0.95].")
    return err


def load_thresholds(path: Path | None = None, project_root: str | None = None) -> EvaluationThresholds:
    """Carrega JSON; se ficheiro não existir, devolve defaults e não escreve disco."""
    p = path or default_thresholds_path(project_root)
    if not p.is_file():
        return EvaluationThresholds()
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return EvaluationThresholds()
    return thresholds_from_dict(raw)


def save_thresholds(t: EvaluationThresholds, path: Path | None = None, project_root: str | None = None) -> None:
    """Grava JSON indentado (utf-8)."""
    p = path or default_thresholds_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(t)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
