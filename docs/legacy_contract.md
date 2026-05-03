# Contrato legado — Surf Detection (auditoria Task A)

Documento de referência para **não quebrar** clientes ou automações que assumem o comportamento atual ao refatorar o pipeline (YOLO → RT-DETR + RTMPose, etc.).

**Data da auditoria:** 2026-05-02  
**Versão do app auditado:** monólito em `app.py`, dependências em `requirements.txt`.

**Atualização:** o pipeline de visão evolui (ex.: RT-DETR + filtros COCO para surf); o **contrato HTTP Gradio** descrito abaixo permanece o critério de compatibilidade.

---

## 1. Divergência em relação ao PRD colado

O PRD menciona UI **Tkinter** e `PROJECT_RULES.md`; neste repositório **não há** `PROJECT_RULES.md` e a interface é **Gradio 4.7.1** (web), não Tkinter.

| Aspecto | Estado atual no repo |
|--------|----------------------|
| Frontend | Gradio (`gr.Blocks`), streaming de imagem no navegador |
| Servidor HTTP | Embutido no Gradio (FastAPI/Starlette por baixo) |
| Pipeline visão | Ultralytics **YOLO** (`yolov8s-pose.pt`, `yolov8n.pt`, `yolov8n-seg.pt`) |
| Entrada vídeo | `cv2.VideoCapture` (webcam USB), índices 0–9 |

Qualquer migração para Tkinter seria **mudança de produto**, não apenas troca de modelo; o contrato HTTP abaixo permanece válido enquanto o processo continuar expondo o mesmo servidor Gradio na mesma porta.

---

## 2. Bootstrap do servidor

Trecho relevante em `app.py`:

- `demo.queue().launch(server_name="0.0.0.0", server_port=<porta>, share=False, show_error=True)`
- **Host:** `0.0.0.0` (aceita conexões na rede local, não só localhost).
- **Porta padrão:** `7860`.
- **Override (testes / CI):** variável de ambiente `GRADIO_SERVER_PORT` (inteiro). Se ausente, usa `7860`.

URL típica para uso local: `http://localhost:7860/` ou `http://127.0.0.1:7860/`.

---

## 3. Endpoints HTTP estáveis (contrato mínimo)

Estes paths são expostos pelo **Gradio 4.7.1** com `Blocks().queue().launch(...)`. Clientes ou testes devem continuar obtendo **HTTP 200** e tipos de conteúdo compatíveis após refatorações que mantêm Gradio.

### 3.1 `GET /`

| Campo | Valor esperado |
|-------|----------------|
| Status | `200` |
| Content-Type | `text/html; charset=utf-8` |
| Corpo | HTML da SPA Gradio (inclui assets estáticos referenciados pelo bundle) |

**Uso:** página principal da aplicação no navegador.

### 3.2 `GET /info`

| Campo | Valor esperado |
|-------|----------------|
| Status | `200` |
| Content-Type | `application/json` |
| Corpo | JSON com metadados da API Gradio (endpoints nomeados/anônimos, etc.) |

Exemplo ilustrativo (app vazio; o app real pode ter mais chaves):

```json
{
  "named_endpoints": {},
  "unnamed_endpoints": {}
}
```

### 3.3 `GET /config`

| Campo | Valor esperado |
|-------|----------------|
| Status | `200` |
| Content-Type | `application/json` |
| Corpo | JSON de configuração da interface (componentes, temas, etc.) |

**Nota:** o payload é grande e pode mudar levemente entre versões do Gradio; testes devem validar **status + JSON parseável**, não um snapshot byte-a-byte.

### 3.4 `GET /openapi.json` (opcional)

Em Gradio **4.7.1** com o `Blocks` atual, este path pode responder **`500`** por limitações do gerador OpenAPI. **Não** use como gate de compatibilidade CI; prefira `/`, `/info` e `/config`.

### 3.5 Streaming / fila (UI em tempo real)

A atualização contínua do vídeo usa a **fila Gradio**, não um único GET estático:

| Path | Método | Papel |
|------|--------|--------|
| `/queue/join` | POST | Cliente entra na fila para funções com streaming/`every` |
| `/queue/data` | GET | SSE/stream de eventos da fila |
| `/queue/status` | GET | Estado da fila |

**Contrato:** enquanto a UI usar `demo.load(..., every=...)`, o navegador continuará dependendo desses paths. Refatorações não devem remover `queue()` ou substituir o servidor sem equivalente documentado.

---

## 4. Sem REST customizado no app atual

Não há Flask/FastAPI declarado em `app.py` além do stack interno do Gradio. Não existem paths tipo `/api/detect` próprios neste repositório — o “endpoint exposto” é **o servidor Gradio completo** na porta configurada.

---

## 5. Payloads de negócio (app atual)

- **Não há** API JSON documentada para “lista de detecções por frame” exposta como REST público.
- Detecções são renderizadas no frame (OpenCV / `result.plot()` Ultralytics) e enviadas à UI como imagem via Gradio.

Para a POC nova (manobras, scores), novos endpoints **podem** ser adicionados desde que os paths acima continuem respondendo como hoje.

---

## 6. Pipeline YOLO (mapa rápido)

| Área | Local |
|------|--------|
| Modelos | `MODELS` dict ~linha 94 em `app.py` |
| Carregamento | `load_model()`, Ultralytics `YOLO(path)` |
| Inferência frame | `process_frame()` → `model(frame, ...)` |
| Pose | chave `'pose'`, tipo `'pose'`, `yolov8s-pose.pt` |
| Objetos COCO | `'objects'`, `'boxes'`, etc., `yolov8n.pt` |
| Segmentação | `'streets'`, `'sidewalks'`, `yolov8n-seg.pt` |

---

## 7. Checklist de compatibilidade para PRs futuros

- [ ] `GET /` → 200, HTML Gradio.
- [ ] `GET /info` → 200, JSON válido.
- [ ] `GET /config` → 200, JSON válido.
- [ ] Porta padrão **7860** quando `GRADIO_SERVER_PORT` não está definida.
- [ ] `server_name` continua bind em `0.0.0.0` (ou documentar mudança).

Teste automatizado: `tests/functional/test_legacy_http_contract.py` (marcador `integration`).
