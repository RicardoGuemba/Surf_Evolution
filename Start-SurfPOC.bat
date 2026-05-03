@echo off
REM Duplo clique no Windows — inicia o Gradio na pasta deste arquivo
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python nao encontrado no PATH. Instale Python 3.10+ e marque "Add to PATH".
  pause
  exit /b 1
)

echo Verificando dependencias...
python -c "import gradio,cv2,ultralytics" 2>nul
if errorlevel 1 (
  echo Instalando requirements.txt ...
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Falha na instalacao.
    pause
    exit /b 1
  )
)

mkdir recordings 2>nul
echo.
echo Abrindo http://localhost:7860 em seguida...
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:7860"

python app.py
echo.
pause
