import cv2
import json
import numpy as np
import gradio as gr
import os
from datetime import datetime
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from surf_app.evaluation_performance import (
    PerformanceEvaluationRuntime,
    format_avaliacao_markdown,
)
from surf_app.evaluation_thresholds import (
    load_thresholds,
    save_thresholds,
    thresholds_from_dict,
    validate_thresholds,
)
from surf_app.poc_maneuvers import SurfPocManeuverEngine

# Configura PyTorch para carregar modelos YOLO (compatibilidade PyTorch 2.6+)
import torch

# Configuração de logging (antes de usar logger)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Otimizações para M4 Silicon Mac
if torch.backends.mps.is_available():
    # Usa Metal Performance Shaders (MPS) no M4
    device = torch.device("mps")
    try:
        torch.mps.empty_cache()  # Limpa cache do MPS
    except AttributeError:
        # Versões antigas do PyTorch podem não ter este método
        pass
    logger.info("✅ MPS (Metal) disponível - usando aceleração GPU do M4")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("✅ CUDA disponível")
else:
    device = torch.device("cpu")
    logger.info("⚠️ Usando CPU (sem aceleração GPU)")

# Otimizações de performance (CPU tipo i7: usa até 8 threads OpenMP do PyTorch)
_cpu_n = os.cpu_count() or 4
torch.set_num_threads(min(8, max(1, _cpu_n)))
# MPS não tem allow_tf32, apenas CUDA
if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'allow_tf32'):
    torch.backends.cuda.allow_tf32 = True

# Para PyTorch 2.6+, configura para permitir carregamento de modelos YOLO
# Monkey-patch torch.load para usar weights_only=False por padrão (apenas para modelos locais confiáveis)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patch para torch.load que desabilita weights_only para modelos YOLO"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Importa YOLO após configurar torch.load
from ultralytics import YOLO

# Restaura torch.load original após importar (mas o patch já foi aplicado)
# O Ultralytics já tem o patch aplicado internamente

# Porta HTTP do Gradio (padrão 7860; testes podem definir GRADIO_SERVER_PORT)
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))


def _resolve_inference_imgsize() -> int:
    """Menor imgsz em CPU/MPS reduz latência; CUDA mantém 640 por padrão."""
    raw = os.environ.get("INFERENCE_IMG_SIZE", "").strip()
    if raw.isdigit():
        return max(320, min(960, int(raw)))
    if torch.cuda.is_available():
        return 640
    return 512


INFERENCE_IMG_SIZE = _resolve_inference_imgsize()

# Estados globais
camera = None
is_recording = False
video_writer = None
output_file = None
lock = threading.Lock()

# Estatísticas de detecção
detection_stats = defaultdict(int)  # Contador de objetos por classe
detection_history = []  # Histórico de detecções por frame
stats_start_time = None  # Tempo de início da coleta de estatísticas
stats_enabled = False  # Flag para habilitar/desabilitar coleta de estatísticas

# Sistema de tracking para contar objetos por aparição (não por frame)
tracked_objects = {}  # {track_id: {'class': str, 'bbox': tuple, 'last_seen': int, 'counted': bool}}
next_track_id = 1
frame_counter = 0
TRACKING_THRESHOLD = 0.3  # IoU mínimo para considerar o mesmo objeto
MAX_FRAMES_WITHOUT_UPDATE = 10  # Frames sem atualização antes de remover do tracking (aumentado para M4)

# Configurações para reduzir falsos positivos
CONFIDENCE_THRESHOLD = 0.5  # Threshold de confiança aumentado para reduzir falsos positivos
MIN_BBOX_AREA = 100  # Área mínima de bounding box (em pixels²) para considerar detecção válida
NMS_IOU_THRESHOLD = 0.5  # IoU threshold para Non-Maximum Suppression entre modelos diferentes

CAPTURE_MAX_FRAME_H = 720
CAPTURE_MAX_FRAME_W = 1280

# (largura, altura) dos frames gravados — iguais ao pipeline de captura (evita MP4 corrompido)
recording_target_size: tuple[int, int] | None = None


def downscale_frame_bgr(frame: np.ndarray) -> np.ndarray:
    """Mesmo redimensionamento usado em capture_frame antes da inferência."""
    if frame.shape[0] > CAPTURE_MAX_FRAME_H or frame.shape[1] > CAPTURE_MAX_FRAME_W:
        scale = min(
            CAPTURE_MAX_FRAME_H / frame.shape[0],
            CAPTURE_MAX_FRAME_W / frame.shape[1],
        )
        nw = int(frame.shape[1] * scale)
        nh = int(frame.shape[0] * scale)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return frame


def recording_dimensions_even(width: int, height: int) -> tuple[int, int]:
    """H.264 / MP4 costumam exigir dimensões pares."""
    w = max(2, width - (width % 2))
    h = max(2, height - (height % 2))
    return w, h


# Carregar modelos — alvo: inferência viável em CPU (ex.: Intel Core i7 sem GPU dedicada)
# Pose: nano é bem mais barato que 's'; Ultralytics baixa o .pt se não existir localmente.
_pose_local = os.path.join(os.path.dirname(__file__), 'yolov8n-pose.pt')
POSE_MODEL_PATH = _pose_local if os.path.exists(_pose_local) else 'yolov8n-pose.pt'

MODELS = {
    'pose': {
        'path': POSE_MODEL_PATH,
        'type': 'pose',
        'model': None,
        'enabled': True,
    },
    # YOLOv8n: menor latência em Mac (CPU/MPS) que RT-DETR; mesma API Ultralytics
    'surf': {
        'path': 'yolov8n.pt',
        'type': 'detection',
        'model': None,
        'enabled': True,
        'labels_allowlist': frozenset({'person', 'surfboard'}),
        'pair_overlay': True,
    },
}

RULES_PATH = os.path.join(os.path.dirname(__file__), "rules", "maneuvers.rules.json")
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

inference_active = False
inference_paused = False
last_bgr_processed = None
last_features_display: dict = {}
pose_prev_state: dict = {}
last_pose_monotonic = None
score_total = 0.0
maneuver_events: list = []

poc_maneuver_engine = SurfPocManeuverEngine()
eval_runtime = PerformanceEvaluationRuntime()
try:
    eval_runtime.thresholds = load_thresholds(project_root=_PROJECT_ROOT)
except Exception as _eval_load_err:
    logger.warning("Carregar evaluation_performance.json: %s", _eval_load_err)


def format_avaliacao_app_md() -> str:
    return format_avaliacao_markdown(eval_runtime)


def _eval_json_text() -> str:
    t = load_thresholds(project_root=_PROJECT_ROOT)
    return json.dumps(asdict(t), indent=2, ensure_ascii=False)


def on_eval_apply(json_str: str) -> tuple[str, str, str]:
    try:
        d = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"JSON inválido: {e}", json_str, format_avaliacao_app_md()
    if not isinstance(d, dict):
        return "A raiz do JSON deve ser um objeto.", json_str, format_avaliacao_app_md()
    t = thresholds_from_dict(d)
    errs = validate_thresholds(t)
    if errs:
        return " · ".join(errs), json_str, format_avaliacao_app_md()
    eval_runtime.thresholds = t
    eval_runtime.reset()
    pretty = json.dumps(asdict(t), indent=2, ensure_ascii=False)
    return "OK — aplicado à sessão (FSM de avaliação reiniciado).", pretty, format_avaliacao_app_md()


def on_eval_save(json_str: str) -> tuple[str, str, str]:
    try:
        d = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"JSON inválido: {e}", json_str, format_avaliacao_app_md()
    if not isinstance(d, dict):
        return "A raiz do JSON deve ser um objeto.", json_str, format_avaliacao_app_md()
    t = thresholds_from_dict(d)
    errs = validate_thresholds(t)
    if errs:
        return " · ".join(errs), json_str, format_avaliacao_app_md()
    save_thresholds(t, project_root=_PROJECT_ROOT)
    eval_runtime.thresholds = t
    eval_runtime.reset()
    pretty = json.dumps(asdict(t), indent=2, ensure_ascii=False)
    return "OK — gravado em rules/evaluation_performance.json.", pretty, format_avaliacao_app_md()


def on_eval_reload() -> tuple[str, str, str]:
    t = load_thresholds(project_root=_PROJECT_ROOT)
    eval_runtime.thresholds = t
    eval_runtime.reset()
    pretty = json.dumps(asdict(t), indent=2, ensure_ascii=False)
    return "OK — recarregado do ficheiro.", pretty, format_avaliacao_app_md()


def _default_rules_text() -> str:
    try:
        with open(RULES_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return (
            '{"version":"1.0.0","sampling_fps_assumption":30,'
            '"global":{"min_keypoint_confidence":0.25},"maneuvers":[]}'
        )


RULES_JSON_TEXT = _default_rules_text()

try:
    from surf_app.rules_engine import RulesRuntime, parse_rules_json

    rules_runtime = RulesRuntime(parse_rules_json(RULES_JSON_TEXT))
except Exception as _rules_err:
    logger.warning("Regras iniciais: %s", _rules_err)
    rules_runtime = None


def _draw_pose_keypoints_green_bgr(
    img: np.ndarray,
    result: object,
    *,
    min_conf: float,
) -> None:
    """Apenas pontos-chave em verde (BGR); sem skeleton, caixas nem labels."""
    kpts_obj = getattr(result, "keypoints", None)
    if kpts_obj is None or not hasattr(kpts_obj, "data") or len(kpts_obj.data) == 0:
        return
    h, w = img.shape[:2]
    radius = max(3, min(7, int(min(h, w) / 140)))
    green = (0, 255, 0)
    for pers in kpts_obj.data:
        arr = pers.cpu().numpy()
        for i in range(arr.shape[0]):
            x, y, c = float(arr[i][0]), float(arr[i][1]), float(arr[i][2])
            if c < min_conf:
                continue
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
            cv2.circle(img, (xi, yi), radius, green, -1, lineType=cv2.LINE_AA)


def _draw_rec_badge_bgr(img: np.ndarray) -> None:
    """Indica no frame que o stream está a ser gravado em ficheiro."""
    cv2.circle(img, (14, 14), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        "REC",
        (26, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def load_model(model_key):
    """Carrega um modelo YOLO com otimizações para M4"""
    try:
        model_info = MODELS[model_key]
        if model_info['model'] is None:
            if os.path.exists(model_info['path']):
                logger.info(f"Carregando modelo {model_key} de {model_info['path']}")
                model_info['model'] = YOLO(model_info['path'])
            else:
                logger.info(f"Baixando modelo {model_key}...")
                model_info['model'] = YOLO(model_info['path'])
            
            # Otimizações para M4 Silicon
            if torch.backends.mps.is_available():
                # Mover modelo para MPS se disponível
                try:
                    # YOLO gerencia o device internamente, mas podemos otimizar
                    model_info['model'].to(device)
                except:
                    pass  # Alguns modelos não suportam .to() diretamente
            
            # Otimizações de inferência
            if hasattr(model_info['model'], 'fuse'):
                try:
                    model_info['model'].fuse()  # Funde camadas para inferência mais rápida
                except:
                    pass
            
            logger.info(f"Modelo {model_key} carregado com sucesso (device: {device})")
        return model_info['model']
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {model_key}: {str(e)}")
        return None

def calculate_iou(bbox1, bbox2):
    """Calcula Intersection over Union (IoU) entre duas bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Área de interseção
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Área de união
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_bbox_area(bbox):
    """Calcula a área de uma bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def apply_nms(detections, iou_threshold=0.5):
    """Aplica Non-Maximum Suppression para remover detecções sobrepostas"""
    if not detections:
        return []
    
    # Ordena por confiança (maior primeiro)
    sorted_detections = sorted(detections, key=lambda x: x.get('conf', 0.5), reverse=True)
    
    kept = []
    while sorted_detections:
        # Pega a detecção com maior confiança
        best = sorted_detections.pop(0)
        kept.append(best)
        
        # Remove detecções que se sobrepõem muito com a melhor
        remaining = []
        for det in sorted_detections:
            iou = calculate_iou(best['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)
        sorted_detections = remaining
    
    return kept

def filter_detections_by_area(detections, min_area=100):
    """Filtra detecções com área muito pequena (provavelmente falsos positivos)"""
    filtered = []
    for det in detections:
        area = calculate_bbox_area(det['bbox'])
        if area >= min_area:
            filtered.append(det)
    return filtered

def track_objects(detections, class_name):
    """Rastreia objetos entre frames e conta apenas novas aparições (UMA VEZ por objeto)
    
    IMPORTANTE: Cada objeto é contado APENAS UMA VEZ quando aparece pela primeira vez na tela.
    Objetos que já foram rastreados não são contados novamente, mesmo que apareçam em frames subsequentes.
    """
    global tracked_objects, next_track_id, detection_stats, stats_enabled, frame_counter
    
    if not stats_enabled or not detections:
        return
    
    current_detections = []
    
    # Processa detecções do frame atual
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Valida bbox
        if x2 > x1 and y2 > y1:
            bbox = (float(x1), float(y1), float(x2), float(y2))
            current_detections.append({
                'bbox': bbox,
                'class': class_name
            })
    
    if not current_detections:
        return
    
    # Remove objetos antigos que não foram vistos há muito tempo (apenas desta classe)
    tracks_to_remove = []
    for track_id, track_info in list(tracked_objects.items()):
        if track_info['class'] == class_name:
            frames_since_seen = frame_counter - track_info['last_seen']
            if frames_since_seen > MAX_FRAMES_WITHOUT_UPDATE:
                tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        del tracked_objects[track_id]
    
    # Tenta associar detecções atuais com tracks existentes (apenas desta classe)
    used_tracks = set()
    used_detections = set()
    
    # Primeiro, tenta associar com tracks existentes
    for det_idx, det in enumerate(current_detections):
        if det_idx in used_detections:
            continue
            
        best_iou = 0
        best_track_id = None
        
        for track_id, track_info in tracked_objects.items():
            if track_info['class'] == class_name and track_id not in used_tracks:
                iou = calculate_iou(det['bbox'], track_info['bbox'])
                if iou > best_iou and iou >= TRACKING_THRESHOLD:
                    best_iou = iou
                    best_track_id = track_id
        
        if best_track_id is not None:
            # Atualiza track existente (objeto JÁ FOI CONTADO, apenas atualiza posição)
            # NÃO incrementa contador - objeto já foi contado na primeira aparição
            tracked_objects[best_track_id]['bbox'] = det['bbox']
            tracked_objects[best_track_id]['last_seen'] = frame_counter
            used_tracks.add(best_track_id)
            used_detections.add(det_idx)
    
    # Processa detecções não associadas (NOVOS objetos - ainda não foram contados)
    for det_idx, det in enumerate(current_detections):
        if det_idx in used_detections:
            continue
        
        # NOVA detecção - cria novo track e conta UMA VEZ
        track_id = next_track_id
        next_track_id += 1
        tracked_objects[track_id] = {
            'class': class_name,
            'bbox': det['bbox'],
            'last_seen': frame_counter,
            'counted': True  # Marca como contado imediatamente
        }
        
        # Conta APENAS UMA VEZ quando o objeto aparece pela primeira vez
        detection_stats[class_name] += 1
        logger.info(f"✅ Novo objeto detectado: {class_name} (track_id: {track_id}, total: {detection_stats[class_name]})")

def initialize_camera():
    """Inicializa a câmera USB"""
    global camera
    
    # Se já existe uma câmera aberta, verifica se ainda está funcionando
    if camera is not None and camera.isOpened():
        ret, _ = camera.read()
        if ret:
            logger.info("Câmera já está conectada e funcionando")
            return True
        else:
            # Câmera não está mais funcionando, fecha e tenta reconectar
            camera.release()
            camera = None
    
    try:
        # Tenta diferentes índices de câmera (0, 1, 2...)
        for idx in range(10):
            test_camera = cv2.VideoCapture(idx)
            if test_camera.isOpened():
                ret, frame = test_camera.read()
                if ret and frame is not None:
                    test_camera.release()
                    camera = cv2.VideoCapture(idx)
                    # Configura propriedades da câmera
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    # Verifica se realmente conseguiu abrir
                    if camera.isOpened():
                        logger.info(f"Câmera encontrada e inicializada na porta {idx}")
                        return True
                    else:
                        camera = None
                else:
                    test_camera.release()
        logger.error("Nenhuma câmera USB encontrada ou disponível")
        return False
    except Exception as e:
        logger.error(f"Erro ao inicializar câmera: {str(e)}")
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        return False

def process_frame(frame, enabled_models):
    """Processa um frame com os modelos ativos (otimizado para M4)"""
    global detection_stats, detection_history, stats_enabled, stats_start_time, frame_counter
    
    # Incrementa contador de frames (importante para tracking)
    if stats_enabled:
        frame_counter += 1
    
    # Mantém cópia para evitar problemas de referência
    processed_frame = frame.copy()
    
    # Inicia estatísticas automaticamente se ainda não foram iniciadas
    if not stats_enabled:
        detection_stats.clear()
        detection_history.clear()
        frame_counter = 0
        stats_start_time = datetime.now()
        stats_enabled = True
        logger.info("Coleta de estatísticas iniciada automaticamente")
    
    # Processa modelos de pose e segmentação primeiro
    pose_seg_models = [m for m in enabled_models if MODELS.get(m, {}).get('type') in ['pose', 'segmentation']]
    detection_models = [m for m in enabled_models if MODELS.get(m, {}).get('type') == 'detection']
    
    # Frame detections para estatísticas
    frame_detections = {}
    
    # Processa pose e segmentação
    for model_key in pose_seg_models:
        model_info = MODELS.get(model_key)
        if model_info is None or not model_info['enabled']:
            continue
            
        model = load_model(model_key)
        if model is None:
            continue
            
        try:
            # Executa detecção com threshold de confiança aumentado para reduzir falsos positivos
            # Usa half precision (FP16) no M4 para maior velocidade
            use_half = torch.backends.mps.is_available()
            results = model(
                frame,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                half=use_half,
                imgsz=INFERENCE_IMG_SIZE,
            )
            if len(results) > 0:
                result = results[0]
                global score_total, maneuver_events, last_features_display
                global last_pose_monotonic, pose_prev_state
                global inference_active, inference_paused, rules_runtime
                if (
                    inference_active
                    and not inference_paused
                    and getattr(result, "keypoints", None) is not None
                    and len(result.keypoints.data) > 0
                ):
                    import time as _time

                    from surf_app.keypoint_features import features_from_pose_xy

                    best_i = 0
                    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                        best_area = -1.0
                        for bi, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            a = float((x2 - x1) * (y2 - y1))
                            if a > best_area:
                                best_area = a
                                best_i = bi
                    kpts_np = result.keypoints.data[best_i].cpu().numpy()
                    now_m = _time.monotonic()
                    dt_s = 0.033
                    if last_pose_monotonic is not None:
                        dt_s = max(1e-4, now_m - last_pose_monotonic)
                    last_pose_monotonic = now_m
                    min_c = (
                        rules_runtime.global_min_kpt_confidence()
                        if rules_runtime
                        else 0.35
                    )
                    feats = features_from_pose_xy(kpts_np, min_c, pose_prev_state, dt_s)
                    hip = feats.get("_hip_line_angle_deg")
                    pose_prev_state = {}
                    if hip is not None:
                        pose_prev_state["hip_line_angle_deg"] = float(hip)
                    feats_rules = {k: float(v) for k, v in feats.items() if not k.startswith("_")}
                    last_features_display = feats_rules
                    if rules_runtime:
                        for ev in rules_runtime.tick(feats_rules, now_m):
                            maneuver_events.append(
                                {
                                    **ev,
                                    "ts_iso": datetime.now().isoformat(
                                        timespec="seconds"
                                    ),
                                }
                            )
                            score_total += float(ev["score"])

                    poc_maneuver_engine.min_conf = min_c
                    fw, fh = int(frame.shape[1]), int(frame.shape[0])
                    for ev in poc_maneuver_engine.step(kpts_np, (fw, fh), now_m):
                        maneuver_events.append(
                            {
                                **ev,
                                "ts_iso": datetime.now().isoformat(timespec="seconds"),
                            }
                        )
                        score_total += float(ev["score"])

                    eval_runtime.min_conf = min_c
                    eval_runtime.step(kpts_np, (fw, fh), now_m)

                kpt_draw_min = (
                    float(rules_runtime.global_min_kpt_confidence())
                    if rules_runtime
                    else 0.35
                )
                _draw_pose_keypoints_green_bgr(
                    processed_frame, result, min_conf=kpt_draw_min
                )

                # Prepara detecções para tracking (pose e segmentação)
                if stats_enabled:
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        num_keypoints = len(result.keypoints.data) if len(result.keypoints.data) > 0 else 0
                        if num_keypoints > 0:
                            detection_name = f"{model_key}_pose"
                            pose_detections = []
                            # Para pose, usa bounding box do keypoint se disponível
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                for i, box in enumerate(result.boxes):
                                    if i < num_keypoints:
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        pose_detections.append({
                                            'bbox': (float(x1), float(y1), float(x2), float(y2))
                                        })
                            else:
                                # Se não houver box, cria uma estimativa baseada nos keypoints
                                for i in range(num_keypoints):
                                    kpts = result.keypoints.data[i]
                                    if len(kpts) > 0 and kpts.numel() > 0:
                                        # Filtra keypoints válidos (visibilidade > 0)
                                        valid_kpts = kpts[kpts[:, 2] > 0] if kpts.shape[1] > 2 else kpts
                                        if len(valid_kpts) > 0:
                                            x_coords = valid_kpts[:, 0].cpu().numpy()
                                            y_coords = valid_kpts[:, 1].cpu().numpy()
                                            x1, y1 = float(x_coords.min()), float(y_coords.min())
                                            x2, y2 = float(x_coords.max()), float(y_coords.max())
                                            # Adiciona margem
                                            margin = 10
                                            pose_detections.append({
                                                'bbox': (max(0, x1 - margin), max(0, y1 - margin), 
                                                        x2 + margin, y2 + margin)
                                            })
                            
                            if pose_detections:
                                track_objects(pose_detections, detection_name)
                            frame_detections[detection_name] = num_keypoints
                            logger.debug("Modelo %s: %s pose(s)", model_key, num_keypoints)
                    elif hasattr(result, 'boxes') and result.boxes is not None:
                        num_boxes = len(result.boxes)
                        if num_boxes > 0:
                            detection_name = f"{model_key}_object"
                            seg_detections = []
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                seg_detections.append({
                                    'bbox': (float(x1), float(y1), float(x2), float(y2))
                                })
                            if seg_detections:
                                track_objects(seg_detections, detection_name)
                            frame_detections[detection_name] = num_boxes
                            logger.info(f"✅ Modelo {model_key} detectou {num_boxes} objeto(s)")
                else:
                    # Log sem estatísticas
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        num_keypoints = len(result.keypoints.data) if len(result.keypoints.data) > 0 else 0
                        if num_keypoints > 0:
                            logger.debug("Modelo %s: %s pose(s)", model_key, num_keypoints)
                    elif hasattr(result, 'boxes') and result.boxes is not None:
                        num_boxes = len(result.boxes)
                        if num_boxes > 0:
                            logger.debug("Modelo %s: %s objeto(s)", model_key, num_boxes)
        except Exception as e:
            logger.error(f"Erro ao processar frame com modelo {model_key}: {str(e)}")
            continue
    
    # Inicializa dicionário para tracking
    detections_for_tracking = defaultdict(list)
    
    # Processa modelos de detecção
    for model_key in detection_models:
        model_info = MODELS.get(model_key)
        if model_info is None or not model_info['enabled']:
            continue
            
        model = load_model(model_key)
        if model is None:
            continue
            
        try:
            # Otimizações para M4: FP16 quando suportado (RT-DETR pode falhar em half — fallback)
            use_half = torch.backends.mps.is_available()
            try:
                results = model(
                    frame,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    half=use_half,
                    imgsz=INFERENCE_IMG_SIZE,
                )
            except Exception:
                results = model(
                    frame,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    half=False,
                    imgsz=INFERENCE_IMG_SIZE,
                )
            # Contador de classes para este modelo
            class_counts = defaultdict(int)
            
            # Coleta todas as detecções primeiro para aplicar filtros
            all_detections = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Valida confiança (já filtrado pelo YOLO, mas verifica novamente)
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Valida tamanho da bounding box
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area < MIN_BBOX_AREA:
                        continue  # Ignora detecções muito pequenas
                    
                    if hasattr(model, 'names') and cls in model.names:
                        label = model.names[cls]
                    else:
                        label = f'Class {cls}'

                    allowlist = model_info.get('labels_allowlist')
                    if allowlist is not None and label not in allowlist:
                        continue

                    all_detections.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'conf': conf,
                        'label': label,
                        'model': model_key
                    })
            
            # Aplica NMS para remover detecções sobrepostas do mesmo modelo
            filtered_detections = apply_nms(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
            
            # Filtra por área mínima novamente (após NMS)
            filtered_detections = filter_detections_by_area(filtered_detections, min_area=MIN_BBOX_AREA)
            
            # Processa apenas as detecções filtradas
            for det in filtered_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                label = det['label']
                
                # Prepara detecção para tracking
                if stats_enabled:
                    full_label = f"{model_key}_{label}" if len(detection_models) > 1 else label
                    class_counts[label] = class_counts.get(label, 0) + 1
                    # Armazena bbox para tracking
                    detections_for_tracking[full_label].append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'conf': conf
                    })

            # Adiciona contagens do frame ao histórico
            if stats_enabled and class_counts:
                frame_detections.update(class_counts)
                                  
        except Exception as e:
            logger.error(f"Erro ao processar frame com modelo {model_key}: {str(e)}")
            continue
    
    # NMS apenas dentro da mesma classe — person e surfboard sobrepõem-se no vídeo de surf
    if stats_enabled and detections_for_tracking:
        detections_for_tracking_filtered = defaultdict(list)
        for class_name, detections in detections_for_tracking.items():
            as_list = [
                {'bbox': d['bbox'], 'conf': d.get('conf', 0.5)} for d in detections
            ]
            filtered_same = apply_nms(as_list, iou_threshold=NMS_IOU_THRESHOLD)
            for det in filtered_same:
                detections_for_tracking_filtered[class_name].append({
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                })
        detections_for_tracking = detections_for_tracking_filtered
    
    # Aplica tracking para contar objetos por aparição (não por frame)
    # IMPORTANTE: Cada objeto é contado APENAS UMA VEZ quando aparece pela primeira vez
    if stats_enabled and detections_for_tracking:
        for class_name, detections in detections_for_tracking.items():
            track_objects(detections, class_name)
    
    # Adiciona detecções do frame ao histórico
    if stats_enabled and frame_detections:
        detection_history.append({
            'timestamp': datetime.now(),
            'detections': frame_detections.copy()
        })
    
    # Limpa cache do MPS periodicamente (a cada 100 frames) para evitar vazamento de memória
    if torch.backends.mps.is_available() and frame_counter % 100 == 0:
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass  # Versões antigas do PyTorch podem não ter este método

    return processed_frame

def capture_frame():
    """Captura um frame da câmera e processa (otimizado para M4)"""
    global camera, is_recording, video_writer, output_file
    global last_bgr_processed, inference_active, inference_paused
    
    # Verifica e inicializa câmera se necessário
    if camera is None or not camera.isOpened():
        if not initialize_camera():
            # Retorna frame de placeholder (já em RGB)
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (24, 24, 28)
            # Converte para RGB
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return placeholder_rgb
    
    # Captura frame da câmera
    try:
        ret, frame = camera.read()
        if not ret or frame is None:
            # Tenta reinicializar a câmera uma vez
            logger.warning("Erro ao ler frame, tentando reinicializar câmera...")
            if initialize_camera():
                ret, frame = camera.read()
            
            if not ret or frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                placeholder[:] = (24, 24, 28)
                # Converte para RGB
                placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                return placeholder_rgb
    except Exception as e:
        logger.error(f"Exceção ao capturar frame: {str(e)}")
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (24, 24, 28)
        # Converte para RGB
        placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        return placeholder_rgb
    
    frame = downscale_frame_bgr(frame)

    enabled_models = [key for key, info in MODELS.items() if info.get("enabled", False)]

    if not inference_active:
        processed_frame = frame
    elif inference_paused:
        if last_bgr_processed is not None:
            processed_frame = last_bgr_processed.copy()
        else:
            processed_frame = frame.copy()
    elif enabled_models:
        try:
            processed_frame = process_frame(frame, enabled_models)
        except Exception as e:
            logger.error(f"Erro ao processar frame: {str(e)}")
            processed_frame = frame
    else:
        processed_frame = frame

    if inference_active and not inference_paused:
        last_bgr_processed = processed_frame.copy()

    # Grava o frame se estiver gravando (em BGR para o VideoWriter)
    if is_recording and video_writer is not None:
        try:
            if video_writer.isOpened():
                tw, th = recording_target_size or (
                    processed_frame.shape[1],
                    processed_frame.shape[0],
                )
                if (
                    processed_frame.shape[1] != tw
                    or processed_frame.shape[0] != th
                ):
                    processed_frame = cv2.resize(
                        processed_frame,
                        (tw, th),
                        interpolation=cv2.INTER_LINEAR,
                    )
                video_writer.write(processed_frame)
            else:
                logger.warning("VideoWriter não está aberto, parando gravação")
                is_recording = False
        except Exception as e:
            logger.error(f"Erro ao gravar frame: {str(e)}")
            # Tenta parar a gravação se houver erro persistente
            try:
                if video_writer is not None:
                    video_writer.release()
                is_recording = False
            except:
                pass

    if is_recording:
        _draw_rec_badge_bgr(processed_frame)
    
    # Converte BGR para RGB para exibição no Gradio
    # OpenCV usa BGR, mas Gradio espera RGB
    try:
        if processed_frame is None:
            logger.error("Processed frame é None")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (24, 24, 28)
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return placeholder_rgb
        
        # Garante que é um numpy array
        if not isinstance(processed_frame, np.ndarray):
            logger.error(f"Frame não é um numpy array: {type(processed_frame)}")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (24, 24, 28)
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return placeholder_rgb
        
        # Garante que o dtype é uint8
        if processed_frame.dtype != np.uint8:
            processed_frame = processed_frame.astype(np.uint8)
        
        if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
            # Converte BGR para RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # Garante que os valores estão no range correto [0, 255]
            processed_frame_rgb = np.clip(processed_frame_rgb, 0, 255).astype(np.uint8)
            return processed_frame_rgb
        elif len(processed_frame.shape) == 2:
            # Se for grayscale, converte para RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            processed_frame_rgb = np.clip(processed_frame_rgb, 0, 255).astype(np.uint8)
            return processed_frame_rgb
        else:
            logger.warning(f"Formato de frame inesperado: {processed_frame.shape}")
            # Tenta redimensionar ou converter
            if len(processed_frame.shape) == 3:
                # Pode ser que tenha 4 canais (RGBA), converte para RGB
                if processed_frame.shape[2] == 4:
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGB)
                    return processed_frame_rgb
            # Fallback: retorna frame original (já deve estar em RGB ou será convertido pelo Gradio)
            return processed_frame
    except Exception as e:
        logger.error(f"Erro ao converter frame BGR para RGB: {str(e)}")
        # Tenta retornar um placeholder em caso de erro
        try:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (24, 24, 28)
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return placeholder_rgb
        except:
            # Último recurso: retorna um array numpy simples
            return np.zeros((480, 640, 3), dtype=np.uint8)

def start_recording():
    """Inicia gravação em arquivo .mp4 (codec compatível com container MP4)."""
    global is_recording, video_writer, output_file, camera, recording_target_size

    if is_recording:
        return "❌ Gravação já está ativa"

    if camera is None or not camera.isOpened():
        if not initialize_camera():
            return "❌ Câmera não disponível. Não é possível iniciar gravação."

    try:
        recordings_dir = "recordings"
        os.makedirs(recordings_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sempre ficheiro .mp4 (contentor MP4)
        output_file = os.path.join(recordings_dir, f"video_{timestamp}.mp4")
        root, ext = os.path.splitext(output_file)
        if ext.lower() != ".mp4":
            output_file = root + ".mp4"

        ret, probe = camera.read()
        if not ret or probe is None:
            return "❌ Não foi possível ler um frame para definir o tamanho do vídeo."

        probe = downscale_frame_bgr(probe)
        height, width = probe.shape[0], probe.shape[1]
        width, height = recording_dimensions_even(width, height)
        recording_target_size = (width, height)

        # Codecs para contentor MP4 (.mp4). mp4v primeiro — melhor compatibilidade OpenCV → ficheiro .mp4.
        fourccs = [
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
            ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
            ("H264", cv2.VideoWriter_fourcc(*"H264")),
        ]
        fourcc = None
        codec_name = None
        test_file = os.path.join(recordings_dir, ".codec_probe.mp4")

        for name, codec in fourccs:
            try:
                test_writer = cv2.VideoWriter(
                    test_file, codec, 30.0, (width, height)
                )
                if test_writer.isOpened():
                    test_writer.release()
                    if os.path.exists(test_file):
                        try:
                            os.remove(test_file)
                        except OSError:
                            pass
                    fourcc = codec
                    codec_name = name
                    logger.info("Codec MP4 selecionado: %s", codec_name)
                    break
            except Exception as e:
                logger.debug("Codec %s indisponível: %s", name, e)

        if fourcc is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            codec_name = "mp4v"
            logger.warning("Fallback codec mp4v (MPEG-4 Part 2) para MP4")

        fps = 30.0
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            video_writer = None
            recording_target_size = None
            return f"❌ Não foi possível criar arquivo de vídeo com codec {codec_name}"

        is_recording = True
        logger.info(
            "Gravação MP4: %s (%s, %dx%d @ %.1ffps)",
            output_file,
            codec_name,
            width,
            height,
            fps,
        )
        return (
            f"✅ Gravação .mp4: {os.path.basename(output_file)} "
            f"({codec_name}, {width}x{height})"
        )

    except Exception as e:
        logger.error(f"Erro ao iniciar gravação: {str(e)}")
        recording_target_size = None
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
            video_writer = None
        is_recording = False
        return f"❌ Erro ao iniciar gravação: {str(e)}"


def stop_recording():
    """Para a gravação de vídeo"""
    global is_recording, video_writer, output_file, recording_target_size
    
    if not is_recording:
        return "⚠️ Não há gravação ativa"
    
    saved_file = output_file
    is_recording = False
    
    # Fecha o VideoWriter de forma segura
    if video_writer is not None:
        try:
            if video_writer.isOpened():
                video_writer.release()
            video_writer = None
        except Exception as e:
            logger.error(f"Erro ao fechar VideoWriter: {str(e)}")
            video_writer = None
    
    output_file = None
    recording_target_size = None

    # Verifica se o arquivo foi criado
    if saved_file and os.path.exists(saved_file):
        file_size = os.path.getsize(saved_file)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Gravação finalizada: {saved_file} ({file_size_mb:.2f} MB)")
        return f"✅ Gravação salva: {os.path.basename(saved_file)} ({file_size_mb:.2f} MB)"
    elif saved_file:
        logger.warning(f"Arquivo de gravação não encontrado: {saved_file}")
        return f"⚠️ Gravação parada, mas arquivo não encontrado: {os.path.basename(saved_file)}"
    else:
        logger.warning("Gravação parada, mas nenhum arquivo foi especificado")
        return "⚠️ Gravação parada"


def start_system():
    """Inicia o sistema (câmera e modelos)"""
    global camera, stats_enabled, detection_stats, detection_history, stats_start_time
    global tracked_objects, next_track_id, frame_counter
    
    messages = []
    
    # Inicializa câmera
    if camera is None or not camera.isOpened():
        if initialize_camera():
            messages.append("✅ Câmera inicializada com sucesso")
        else:
            messages.append("❌ Erro ao inicializar câmera")
    else:
        messages.append("✅ Câmera já está conectada")
    
    # Carrega modelos habilitados
    enabled = [key for key, info in MODELS.items() if info.get('enabled', False)]
    for model_key in enabled:
        model = load_model(model_key)
        if model:
            messages.append(f"✅ Modelo '{model_key}' carregado")
        else:
            messages.append(f"❌ Erro ao carregar modelo '{model_key}'")
    
    if not enabled:
        messages.append("⚠️ Nenhum modelo habilitado")
    
    # Inicia coleta de estatísticas automaticamente e reseta tracking
    if not stats_enabled:
        detection_stats.clear()
        detection_history.clear()
        tracked_objects.clear()
        next_track_id = 1
        frame_counter = 0
        stats_start_time = datetime.now()
        stats_enabled = True
        messages.append("📊 Coleta de estatísticas iniciada automaticamente")
    
    return "\n".join(messages) if messages else "✅ Sistema pronto"


# Carrega modelo inicial
load_model('pose')

logger.info(
    "Latência: imgsz=%s (export INFERENCE_IMG_SIZE=416 para mais FPS em Mac)",
    INFERENCE_IMG_SIZE,
)

# Inicializa câmera
initialize_camera()

SURF_APP_TITLE = "Surf Evolution - SURF BRASIL 50 & 60 +"
# Subir de valor quando a UI mudar; no terminal verá esta linha ao iniciar.
SURF_UI_REVISION = "2026-05-02-green-kpts-no-exit"

SURF_HEAD = (
    '<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0" />'
    '<meta http-equiv="Pragma" content="no-cache" />'
)

SURF_CSS = """
.gradio-container {
  background: linear-gradient(165deg, #dbeafe 0%, #e0f2fe 35%, #fff7ed 70%, #ffedd5 100%) !important;
}
footer { display: none !important; }
"""

# Interface Gradio — duas abas: avaliação de performance | pontuação (futuro)
with gr.Blocks(
    title=SURF_APP_TITLE,
    head=SURF_HEAD,
    theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="orange",
        neutral_hue="stone",
        radius_size="md",
        spacing_size="md",
    ),
    css=SURF_CSS,
) as demo:
    gr.Markdown(f"# {SURF_APP_TITLE}")

    with gr.Row():
        sim_start_btn = gr.Button("Iniciar", variant="primary", size="lg")
        sim_pause_btn = gr.Button("Pause", size="lg")
        gravar_btn = gr.Button("Gravar", variant="secondary", size="lg")

    with gr.Tabs():
        with gr.Tab("Avaliação de performance"):
            with gr.Row():
                with gr.Column(scale=3):
                    video_output = gr.Image(
                        show_label=False,
                        height=380,
                        type="numpy",
                        streaming=True,
                    )
                with gr.Column(scale=2):
                    avaliacao_md = gr.Markdown(value=format_avaliacao_app_md())
                    with gr.Accordion(
                        "Parâmetros de avaliação (thresholds e tempos)",
                        open=False,
                    ):
                        gr.Markdown(
                            "Edite o mesmo esquema que `rules/evaluation_performance.json`. "
                            "**Aplicar à sessão** valida e usa os valores na avaliação em tempo real (reinicia o estado da avaliação). "
                            "**Gravar no ficheiro** persiste no projeto; **Recarregar** descarta edições não gravadas e lê o disco."
                        )
                        eval_params_tb = gr.Textbox(
                            label="JSON",
                            lines=22,
                            value=_eval_json_text(),
                        )
                        eval_feedback_tb = gr.Textbox(
                            label="Validação / estado",
                            lines=2,
                            interactive=False,
                        )
                        with gr.Row():
                            eval_apply_btn = gr.Button("Aplicar à sessão", variant="primary")
                            eval_save_btn = gr.Button("Gravar no ficheiro")
                            eval_reload_btn = gr.Button("Recarregar do ficheiro")

        with gr.Tab("Pontuação Surf"):
            gr.Markdown(
                "### Pontuação Surf\n\n"
                "_Módulo reservado: regras e interface de pontuação serão implementadas posteriormente._"
            )

    def on_sim_iniciar():
        """Inicia sessão nova, ou retoma se já estiver pausada."""
        global inference_active, inference_paused, score_total, maneuver_events
        global last_pose_monotonic, pose_prev_state
        if inference_active and not inference_paused:
            return format_avaliacao_app_md()
        if inference_active and inference_paused:
            inference_paused = False
            last_pose_monotonic = None
            return format_avaliacao_app_md()
        inference_active = True
        inference_paused = False
        score_total = 0.0
        maneuver_events = []
        last_pose_monotonic = None
        pose_prev_state = {}
        poc_maneuver_engine.reset()
        eval_runtime.reset()
        if rules_runtime:
            rules_runtime.reset_hold()
        start_system()
        return format_avaliacao_app_md()

    def on_sim_pause():
        global inference_paused
        inference_paused = True
        return format_avaliacao_app_md()

    def on_gravar_toggle():
        if is_recording:
            stop_recording()
        else:
            start_recording()
        return format_avaliacao_app_md()

    def on_load():
        global inference_active, inference_paused
        inference_active = False
        inference_paused = False
        try:
            eval_runtime.thresholds = load_thresholds(project_root=_PROJECT_ROOT)
        except Exception as _e:
            logger.warning("on_load: thresholds: %s", _e)
        eval_runtime.reset()
        start_system()
        return format_avaliacao_app_md(), _eval_json_text(), ""

    demo.load(
        on_load,
        outputs=[avaliacao_md, eval_params_tb, eval_feedback_tb],
    )

    # Menos polls que inferências lentas = fila Gradio menor e sensação de menor atraso
    if torch.cuda.is_available():
        fps_interval = 0.033
    elif torch.backends.mps.is_available():
        fps_interval = 0.04
    else:
        fps_interval = 0.055

    demo.load(
        capture_frame,
        outputs=video_output,
        every=fps_interval,
        show_progress=False,
    )

    demo.load(
        format_avaliacao_app_md,
        outputs=avaliacao_md,
        every=0.35,
        show_progress=False,
    )

    sim_start_btn.click(on_sim_iniciar, outputs=[avaliacao_md])
    sim_pause_btn.click(on_sim_pause, outputs=[avaliacao_md])
    gravar_btn.click(on_gravar_toggle, outputs=[avaliacao_md])

    eval_apply_btn.click(
        on_eval_apply,
        inputs=[eval_params_tb],
        outputs=[eval_feedback_tb, eval_params_tb, avaliacao_md],
    )
    eval_save_btn.click(
        on_eval_save,
        inputs=[eval_params_tb],
        outputs=[eval_feedback_tb, eval_params_tb, avaliacao_md],
    )
    eval_reload_btn.click(
        on_eval_reload,
        outputs=[eval_feedback_tb, eval_params_tb, avaliacao_md],
    )

if __name__ == "__main__":
    import sys

    _cwd = os.getcwd()
    print(
        f"\n{'=' * 60}\n"
        f"{SURF_APP_TITLE}\n"
        f"UI revision: {SURF_UI_REVISION}\n"
        f"Pasta do processo: {_cwd}\n"
        f"Porta: {GRADIO_SERVER_PORT}\n"
        f"{'=' * 60}\n"
        "Se o browser mostrar a UI antiga: pare TODOS os python/app.py, "
        "mate o que usa esta porta, ou use scripts/stop_surf_port.sh\n",
        file=sys.stderr,
        flush=True,
    )
    logger.info("%s | UI revision %s | cwd=%s", SURF_APP_TITLE, SURF_UI_REVISION, _cwd)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
        favicon_path=None,
    )
