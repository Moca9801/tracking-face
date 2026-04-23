# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
)
MODELS_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET_NAME = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface/face_recognition_sface_2021dec.onnx"
CACHE_NAME = ".face_embeddings_cache.pkl"
ENV_MODELS = "FACE_MATCH_MODELS"


def get_models_dir() -> Path:
    override = os.environ.get(ENV_MODELS)
    if override:
        p = Path(override).expanduser().resolve()
    else:
        p = Path.home() / ".cache" / "face_match" / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_model(filename: str) -> Path:
    name = filename.split("/")[-1]
    path = get_models_dir() / name
    if path.is_file() and path.stat().st_size > 1000:
        return path
    url = f"{MODELS_BASE}/{filename}"
    print(f"Descargando modelo: {name} (puede tardar)...", file=sys.stderr)
    part = path.with_suffix(path.suffix + ".part")
    try:
        # Timeout de 60 s de conexión + 300 s de lectura para redes lentas
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            with open(part, "wb") as f:
                while True:
                    chunk = response.read(1 << 16)  # 64 KB por bloque
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo descargar el modelo '{name}'. "
            f"Verifica tu conexión a Internet. Detalle: {exc}"
        ) from exc
    # Validación mínima de integridad: el modelo debe pesar más de 1 MB
    if part.stat().st_size < 1_000_000:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"El archivo descargado '{name}' parece estar corrupto (tamaño inesperadamente pequeño)."
        )
    part.replace(path)
    return path


def load_bgr(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def list_image_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out)


def pick_best_face(faces: np.ndarray | None) -> np.ndarray | None:
    if faces is None or faces.size == 0:
        return None
    f = np.atleast_2d(faces)
    if f.shape[1] < 15:
        return f[0]
    scores = f[:, 14]
    return f[int(np.argmax(scores))]


def embed(
    bgr: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    row = pick_best_face(faces)
    if row is None:
        return None
    aligned = recognizer.alignCrop(bgr, row)
    return recognizer.feature(aligned)


def load_cache(cache_path: Path) -> dict:
    if not cache_path.is_file():
        return {}
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_cache(cache_path: Path, data: dict) -> None:
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    tmp.replace(cache_path)
