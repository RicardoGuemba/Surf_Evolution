"""Testes mínimos do módulo de avaliação de performance."""

from __future__ import annotations

import numpy as np

from surf_app.evaluation_performance import (
    PerformanceEvaluationRuntime,
    _body_axis_x,
    format_avaliacao_markdown,
)


def _fake_kpts() -> np.ndarray:
    """17x3 COCO-like: confiança alta em tronco e pernas."""
    k = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        k[i, 2] = 0.99
    # Ombros e quadril alinhados em x=100
    for idx in (5, 6, 11, 12):
        k[idx, 0] = 100.0
        k[idx, 1] = float(50 + idx)
    # Joelhos ligeiramente afastados do eixo
    k[13, 0] = 108.0
    k[13, 1] = 200.0
    k[14, 0] = 92.0
    k[14, 1] = 200.0
    k[15, 0] = 110.0
    k[15, 1] = 280.0
    k[16, 0] = 90.0
    k[16, 1] = 280.0
    return k


def test_body_axis_x_stable() -> None:
    k = _fake_kpts()
    ax = _body_axis_x(k, 0.5)
    assert ax is not None
    assert abs(ax - 100.0) < 1e-3


def test_format_avaliacao_markdown_runs() -> None:
    rt = PerformanceEvaluationRuntime()
    s = format_avaliacao_markdown(rt)
    assert "Pop-up" in s
    assert "In-pipeline" in s
