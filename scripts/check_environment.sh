#!/usr/bin/env bash
# Verificação rápida do ambiente de desenvolvimento (macOS/Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Surf Detection — verificação de ambiente"
echo "========================================="
echo "Diretório: $ROOT"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERRO: python3 não encontrado."
  exit 1
fi
echo "Python: $(python3 --version)"

echo ""
echo "Dependências Python:"
check_mod () {
  if python3 -c "import $1" 2>/dev/null; then echo "  OK  $1"; else echo "  FALTA  $1"; fi
}
check_mod cv2
check_mod gradio
check_mod ultralytics
check_mod numpy
if python3 -c "from PIL import Image" 2>/dev/null; then echo "  OK  Pillow"; else echo "  FALTA  Pillow"; fi
check_mod matplotlib
check_mod pytest

echo ""
echo "Sintaxe app.py:"
python3 -m py_compile app.py && echo "  OK"

echo ""
echo "Testes unitários (pytest):"
pytest -q tests/unit --tb=no 2>/dev/null || {
  echo "  Falhou ou pytest não instalado."
  exit 1
}
echo ""
echo "Pronto. Subir app: python3 app.py → http://localhost:7860"
