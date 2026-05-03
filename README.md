# Surf Detection — POC

Aplicação monolítica em Python: **Gradio** na porta **7860**, detecção **YOLOv8n** (COCO: `person`, `surfboard`), pose **YOLOv8n-pose**, associação surfista–prancha e overlay em tempo real. Otimizado para **baixa latência em Mac** (CPU/MPS): `imgsz` 512 por padrão fora de CUDA.

## Requisitos

- Python 3.10+
- Webcam USB (opcional para dev sem câmera: o app mostra placeholder)
- Primeira execução pode baixar pesos Ultralytics (internet)

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python3 app.py
```

### Duplo clique (sem terminal manual)

| Sistema | Arquivo |
|---------|---------|
| **macOS** | **`SurfPOC.command`** — duplo clique no Finder; instala deps se faltar e abre o navegador após ~5 s. |
| macOS (atalho) | `Iniciar_Detecção.command` — chama o mesmo script acima. |
| **Windows** | **`Start-SurfPOC.bat`** — duplo clique no Explorer. |

**macOS:** na primeira vez, se o Finder só abrir o editor, clique com o botão direito → **Abrir** e confirme; ou em Terminal: `chmod +x SurfPOC.command`.

Depois acesse **http://localhost:7860** (ou aguarde a aba automática).

**UI não atualiza?** Quase sempre é **outro processo** ainda à escuta na **7860**. Ao iniciar `app.py`, no stderr aparecem **UI revision** e **pasta do processo** — devem corresponder a este repositório (`SURF_UI_REVISION` em `app.py`). Pare o antigo: `bash scripts/stop_surf_port.sh`, suba de novo o app e force reload no browser (**Cmd+Shift+R** / **Ctrl+F5**).

### Variáveis úteis

| Variável | Efeito |
|----------|--------|
| `GRADIO_SERVER_PORT` | Porta HTTP (padrão `7860`). Usada pelos testes de integração. |
| `INFERENCE_IMG_SIZE` | Tamanho de entrada dos modelos (320–960). Em Mac sem CUDA o padrão interno é **512**; use **416** se precisar de mais FPS (troca precisão por velocidade). |

**Nota:** quantização INT8 no PyTorch/MPS ainda não é caminho estável para este stack; **YOLO nano + imgsz menor** costuma dar o melhor ganho no Mac.

## Testes

```bash
pytest                      # unitários (rápido)
pytest -m integration       # contrato HTTP Gradio (sobe o app em subprocess)
bash scripts/check_environment.sh   # deps + sintaxe + unitários
```

`testar_sistema.py` é um smoke test **leve** (apenas imports + presença de `app.py`; não sobe o servidor).

## Estrutura

```
├── app.py                 # UI Gradio + pipeline de captura e inferência
├── surf_app/              # Pacote: avaliação de performance, POC manobras, regras
├── tests/                 # pytest (unit + functional)
├── docs/
│   └── legacy_contract.md # Contrato HTTP legado (Gradio)
├── rules/
│   └── maneuvers.rules.json  # Stub para motor de manobras (evolução)
├── assets/sample_videos/  # Coloque vídeos curtos para testes manuais
├── requirements.txt
├── testar_sistema.py        # smoke test opcional (imports)
├── SurfPOC.command          # macOS: duplo clique para iniciar
├── Iniciar_Detecção.command # macOS: atalho → SurfPOC.command
└── Start-SurfPOC.bat        # Windows: duplo clique
```

## Simulação e regras (POC)

Interface **Surf Evolution - SURF BRASIL 50 & 60 +** (Gradio): abas **Avaliação de performance** (pop-up + in-pipeline em tempo real) e **Pontuação Surf** (placeholder). Botões **Iniciar** / **Pause** / **Sair** / **Gravar**; `session_summary.json` ao sair.

Módulo Python: `surf_app/evaluation_performance.py`. Regras JSON (motor legado): `rules/maneuvers.rules.json`. Features derivadas da pose:  
   `avg_kpt_conf`, `torso_lean_deg`, `hip_rotation_deg_s`, `turn_intensity`, `knee_flex_mean_deg`.  
   Operadores: `>=`, `<=`, `>`, `<`, `==`. Cada condição pode ter `duration_ms`; usa-se o **maior** entre elas como tempo mínimo contínuo.

Arquivo de referência: `rules/maneuvers.rules.json`.

## Modelos (Ultralytics)

| Chave em `MODELS` | Peso | Função |
|-------------------|------|--------|
| `pose` | `yolov8n-pose.pt` | Pose 2D (nano; baixo custo em CPU) |
| `surf` | `yolov8n.pt` | Detecção COCO filtrada a person + surfboard (baixa latência) |

Coloque `yolov8n-pose.pt` na raiz do projeto se quiser modo offline; caso contrário o Ultralytics baixa automaticamente.

## Licença dos pesos / framework

O projeto usa **Ultralytics** e **Gradio**; verifique os termos (ex.: AGPL no ecossistema Ultralytics) antes de uso comercial fechado.

## Histórico resumido

Ver `CHANGELOG.md`.
