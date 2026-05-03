"""
Garante compatibilidade mínima do servidor HTTP legado (Gradio).

Sobe `app.py` em subprocess na porta definida por GRADIO_SERVER_PORT para não
colidir com um dev server na 7860.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PY = PROJECT_ROOT / "app.py"

# Primeira execução pode baixar pesos YOLO; webcam ausente não impede o servidor.
STARTUP_TIMEOUT_S = 240.0
POLL_INTERVAL_S = 2.0


def _http_get_json(url: str, timeout: float = 10.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)


def _http_get_text(url: str, timeout: float = 10.0) -> tuple[int, str, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        ctype = resp.headers.get("Content-Type", "")
        return resp.status, ctype, body


def _wait_until_ready(base: str, deadline: float) -> None:
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            status, _ = _http_get_json(f"{base}/info", timeout=5.0)
            if status == 200:
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"servidor não respondeu a GET /info a tempo: {last_err}")


@pytest.mark.integration
def test_legacy_gradio_http_contract():
    """Paths documentados em docs/legacy_contract.md permanecem válidos."""
    port = int(os.environ.get("GRADIO_SERVER_PORT", "17862"))
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "GRADIO_SERVER_PORT": str(port)}

    proc = subprocess.Popen(
        [sys.executable, str(APP_PY)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + STARTUP_TIMEOUT_S
    try:
        _wait_until_ready(base, deadline)

        st_root, ctype_root, html = _http_get_text(f"{base}/")
        assert st_root == 200
        assert "text/html" in ctype_root.lower()
        assert "gradio" in html.lower()

        st_info, info = _http_get_json(f"{base}/info")
        assert st_info == 200
        assert isinstance(info, dict)

        st_cfg, cfg = _http_get_json(f"{base}/config")
        assert st_cfg == 200
        assert isinstance(cfg, dict)

        # /openapi.json pode falhar em algumas combinações Gradio 4.7 + Blocks;
        # não faz parte do contrato mínimo usado pela UI.
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if proc.returncode not in (0, -15):  # 0 ok; -15 SIGTERM no macOS/Linux
            pytest.fail(f"app subprocess exit code {proc.returncode}")
