"""Lógica reutilizable del detector de manos. No contiene un modelo entrenado de LSP."""
from __future__ import annotations
import datetime as dt
import json
import re
from pathlib import Path
import urllib.request
import uuid
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SEQUENCES = DATA / "sequences"
MODELS = BASE / "models"
EVIDENCE = BASE / "evidencias"
STATIC_DATA = DATA / "static_samples.jsonl"
STATIC_MODEL = MODELS / "static.joblib"
DYNAMIC_MODEL = MODELS / "dynamic.keras"
DYNAMIC_LABELS = MODELS / "dynamic_labels.json"
HAND_MODEL = MODELS / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DIM = 134  # dos manos (126), presencia (2), distancia (2), muñecas absolutas xy (4)
SEQUENCE_LEN = 30
EDGES = (
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
)

def prepare_dirs():
    for directory in (DATA, SEQUENCES, MODELS, EVIDENCE):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_hand_model():
    prepare_dirs()
    if HAND_MODEL.is_file() and HAND_MODEL.stat().st_size > 1_000_000:
        return HAND_MODEL
    tmp = HAND_MODEL.with_suffix(".download")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
        if tmp.stat().st_size < 1_000_000:
            raise ValueError("El archivo recibido no es un modelo válido")
        tmp.replace(HAND_MODEL)
    except Exception as error:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "No se pudo descargar el detector oficial de MediaPipe. Comprueba internet o "
            "descárgalo desde:\n" + HAND_MODEL_URL + "\ny guárdalo como models/hand_landmarker.task.\n" + str(error)
        ) from error
    return HAND_MODEL


def vectorize(result):
    """Vector estático de dos manos. Las posiciones se normalizan por mano."""
    hands = getattr(result, 'hand_landmarks', None) or []
    if not hands:
        return None
    hands = sorted(hands[:2], key=lambda hand: float(hand[0].x))
    rows, wrists, scales = [], [], []
    for hand in hands:
        points = np.asarray([[p.x,p.y,p.z] for p in hand], dtype=np.float32)
        if points.shape != (21,3) or not np.isfinite(points).all():
            return None
        wrist = points[0].copy()
        scale = float(np.linalg.norm(points[9,:2] - wrist[:2]))
        if scale < 1e-6:
            scale = float(np.ptp(points[:,:2], axis=0).max())
        scale = max(scale, 1e-5)
        rows.append(((points - wrist)/scale).reshape(-1))
        wrists.append(wrist)
        scales.append(scale)
    while len(rows) < 2:
        rows.append(np.zeros(63, dtype=np.float32))
    present = np.asarray([1.0, float(len(hands)==2)], dtype=np.float32)
    separation = np.zeros(2, dtype=np.float32)
    if len(hands) == 2:
        separation = (wrists[1][:2] - wrists[0][:2])/max(scales)
    positions = np.zeros(4, dtype=np.float32)
    for index,wrist in enumerate(wrists):
        positions[2*index:2*index+2] = wrist[:2]
    data = np.concatenate([*rows,present,separation,positions]).astype(np.float32)
    assert data.shape == (DIM,)
    return data


def draw_hands(frame, result, cv2):
    height,width = frame.shape[:2]
    hands = getattr(result, 'hand_landmarks', None) or []
    for hand in hands:
        positions = [(int(p.x*width),int(p.y*height)) for p in hand]
        for a,b in EDGES:
            cv2.line(frame,positions[a],positions[b],(0,210,222),2,cv2.LINE_AA)
        for x,y in positions:
            cv2.circle(frame,(x,y),4,(19,99,255),-1,cv2.LINE_AA)
    return len(hands)


def normalize_label(label):
    label = " ".join(label.strip().upper().split())
    if not label or len(label)>32 or not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ0-9 _-]+", label):
        raise ValueError("Escribe una etiqueta de 1 a 32 letras, números, espacios o guiones")
    return label


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def save_static(label, feature, session="", source=""):
    label = normalize_label(label)
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (DIM,) or not np.isfinite(feature).all():
        raise ValueError("Muestra de mano no válida")
    prepare_dirs()
    record={"label":label,"feature":feature.tolist(),"session":session,
            "source":source.strip()[:160],"utc":now_iso()}
    with STATIC_DATA.open('a',encoding='utf-8') as fp:
        fp.write(json.dumps(record,ensure_ascii=False)+"\n")
    return record


def load_static():
    if not STATIC_DATA.exists():
        return []
    rows=[]
    for number,line in enumerate(STATIC_DATA.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        try:
            item=json.loads(line)
            v=np.asarray(item["feature"], dtype=np.float32)
            if v.shape == (DIM,) and np.isfinite(v).all():
                rows.append(item)
        except (ValueError,KeyError,TypeError):
            raise ValueError(f"Hay una muestra dañada en línea {number}")
    return rows


def save_sequence(label, features, session="", source=""):
    label=normalize_label(label)
    seq=np.asarray(features,dtype=np.float32)
    if seq.shape!=(SEQUENCE_LEN,DIM) or not np.isfinite(seq).all():
        raise ValueError("Una secuencia debe contener exactamente 30 fotogramas válidos")
    prepare_dirs()
    # Guardar cada clase como carpeta con nombre seguro; no se usa pickle.
    sub=SEQUENCES/label
    sub.mkdir(parents=True,exist_ok=True)
    target=sub/(uuid.uuid4().hex+'.npz')
    np.savez_compressed(target,frames=seq,session=np.asarray(session),source=np.asarray(source[:160]))
    return target


def load_sequences():
    rows=[]
    for file in SEQUENCES.glob('*/*.npz'):
        with np.load(file,allow_pickle=False) as value:
            mat=value['frames'].astype(np.float32)
            session=str(value['session']) if 'session' in value else ''
        if mat.shape==(SEQUENCE_LEN,DIM) and np.isfinite(mat).all():
            rows.append((file.parent.name,mat,session))
    return rows


def counts(kind):
    from collections import Counter
    return dict(sorted(Counter(r['label'] for r in load_static()).items())) if kind=='static' else \
        dict(sorted(Counter(label for label,_,_ in load_sequences()).items()))
