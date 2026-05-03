# Changelog

## Latência Mac (2026-05)

- Detecção `surf`: **RT-DETR → YOLOv8n** (`yolov8n.pt`).
- Inferência: **`imgsz` 512** quando não há CUDA (640 em CUDA); override via `INFERENCE_IMG_SIZE`.
- Gradio: intervalo de atualização do vídeo ajustado (menos fila em CPU/MPS).

## Consolidação (2026-05)

- Pacote `surf_inference` renomeado para **`surf_app`** (`association.py`).
- README único alinhado ao pipeline atual (RT-DETR ResNet50 + YOLOv8n-pose, Gradio).
- Removido **`GUIA_RAPIDO.md`** (conteúdo absorvido pelo README).
- Scripts shell legados (`test_startup.sh`, `test_sistema_completo.sh`, `test_command_execution.sh`) substituídos por **`scripts/check_environment.sh`**.
- Adicionados **`rules/maneuvers.rules.json`** (stub) e **`assets/sample_videos/`** (placeholder).
- `.gitignore` ajustado para pesos locais opcionais.

---

## Versão simplificada (Gradio)

### Mudanças principais

1. **Removido Flask** — interface apenas Gradio.
2. **Estrutura** — sem `templates/` nem `static/` para o app principal.

### Dependências

- gradio, ultralytics, opencv-python, numpy, Pillow, matplotlib, pytest

### Uso

1. `./Iniciar_Detecção.command` ou `python3 app.py`
2. Navegador: **http://localhost:7860**

### Migração da versão Flask

- Frontend HTML/CSS/JS foi substituído por Gradio em `app.py`.
