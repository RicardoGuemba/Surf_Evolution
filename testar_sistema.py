#!/usr/bin/env python3
"""Smoke test manual: imports e presença de app.py (não inicia o servidor)."""

from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    print("Surf Detection — teste rápido")
    print("=" * 50)

    for label, mod in [
        ("OpenCV", "cv2"),
        ("Gradio", "gradio"),
        ("Ultralytics", "ultralytics"),
        ("NumPy", "numpy"),
    ]:
        try:
            __import__(mod)
            print(f"  OK  {label}")
        except ImportError as e:
            print(f"  ERRO  {label}: {e}")
            return 1

    spec = importlib.util.spec_from_file_location("app_module", os.path.join(ROOT, "app.py"))
    if spec is None:
        print("  ERRO  não foi possível ler app.py")
        return 1
    print("  OK  app.py encontrado")

    print()
    print("Instalar deps: pip install -r requirements.txt")
    print("Subir UI:       python3 app.py → http://localhost:7860")
    return 0


if __name__ == "__main__":
    sys.exit(main())
