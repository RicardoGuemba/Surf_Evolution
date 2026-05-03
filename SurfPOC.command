#!/usr/bin/env bash
#
# Duplo clique no Finder (macOS) para iniciar o Surf POC.
# Requer Terminal.app — o próprio macOS associa .command ao Terminal.
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export PS1='$ '

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este launcher é para macOS (.command). Em Linux use: python3 app.py"
  exit 1
fi

# Se veio do Finder, traz o Terminal para frente
if [[ -z "${TERM_PROGRAM:-}" ]]; then
  osascript -e 'tell application "Terminal" to activate' 2>/dev/null || true
fi

clear
echo "Surf Evolution (Gradio) — launch na pasta do .command"
echo "=============================================="
echo "Pasta: $DIR"
echo ""

if ! command -v python3 &>/dev/null; then
  echo "Python 3 não encontrado. Instale Python 3.10 ou superior."
  echo ""
  read -r -n 1 -s -p "Pressione qualquer tecla para sair..."
  echo ""
  exit 1
fi

echo "Verificando dependências..."
MISSING=false
for py in "import gradio" "import cv2" "import ultralytics"; do
  if ! python3 -c "$py" 2>/dev/null; then
    MISSING=true
    echo "  Falta módulo para: $py"
  fi
done

if [[ "$MISSING" == true ]]; then
  echo ""
  echo "Instalando requirements.txt ..."
  pip3 install -r requirements.txt || {
    echo "Erro ao instalar dependências."
    read -r -n 1 -s -p "Pressione qualquer tecla para sair..."
    echo ""
    exit 1
  }
  echo "Dependências instaladas."
else
  echo "Dependências OK."
fi

mkdir -p recordings

PORT="${GRADIO_SERVER_PORT:-7860}"
echo ""
if command -v lsof >/dev/null 2>&1; then
  OLD=$(lsof -ti ":${PORT}" 2>/dev/null || true)
  if [[ -n "${OLD:-}" ]]; then
    echo "AVISO: porta ${PORT} já em uso (PID: ${OLD})."
    echo "       O browser pode mostrar uma UI ANTIGA. Faça um dos seguintes:"
    echo "       • bash \"$DIR/scripts/stop_surf_port.sh\" ${PORT}"
    echo "       • ou: kill ${OLD}"
    echo ""
  fi
fi
echo "Servidor: http://localhost:${PORT}"
echo "No terminal deve aparecer a linha UI revision: ... ao arrancar."
echo "Ctrl+C para parar."
echo "=============================================="
echo ""

# Abre o navegador alguns segundos depois (melhor esforço — o servidor pode demorar na 1ª vez)
(sleep 5 && open "http://127.0.0.1:${PORT}" 2>/dev/null) &

python3 app.py

echo ""
echo "Servidor encerrado."
read -r -n 1 -s -p "Pressione qualquer tecla para fechar..."
echo ""
