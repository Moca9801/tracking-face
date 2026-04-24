# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
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

# Official SHA256 hashes to ensure integrity and security
MODEL_HASHES = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}

CACHE_NAME = ".face_embeddings_cache.json"
ENV_MODELS = "FACE_MATCH_MODELS"


def _calculate_sha256(path: Path) -> str:
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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
    expected_hash = MODEL_HASHES.get(name)

    # If the file exists, verify integrity before using it
    if path.is_file():
        if expected_hash and _calculate_sha256(path) == expected_hash:
            return path
        else:
            print(f"Model '{name}' corrupt or old. Redownloading...", file=sys.stderr)

    url = f"{MODELS_BASE}/{filename}"
    print(f"Downloading model: {name} (this may take a while)...", file=sys.stderr)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            with open(part, "wb") as f:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Error downloading '{name}': {exc}") from exc

    # Hash integrity verification
    if expected_hash:
        actual_hash = _calculate_sha256(part)
        if actual_hash != expected_hash:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Integrity error in '{name}': Hash mismatch. "
                "The download might have been intercepted or is incomplete."
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
        return f[0]  # type: ignore[no-any-return]
    scores = f[:, 14]
    return f[int(np.argmax(scores))]  # type: ignore[no-any-return]


def embed(
    bgr: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    faces = detector.detect(bgr)
    if faces[1] is not None:
        face = faces[1][0]
        aligned = recognizer.alignCrop(bgr, face)
        feat = recognizer.feature(aligned)
        return feat

    return None


def load_cache(cache_path: Path) -> dict[str, tuple[float, np.ndarray]]:
    """Load the embeddings cache from a JSON file safely."""
    if not cache_path.is_file():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert lists back to numpy arrays
            return {
                k: (v[0], np.array(v[1], dtype=np.float32))
                for k, v in data.items()
            }
    except Exception:
        return {}


def save_cache(cache_path: Path, data: dict[str, tuple[float, np.ndarray]]) -> None:
    """Save the embeddings cache to a JSON file, converting arrays to lists."""
    tmp = cache_path.with_suffix(".tmp")
    # Convert numpy arrays to lists for JSON serialization
    serializable = {
        k: (v[0], v[1].tolist())
        for k, v in data.items()
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    tmp.replace(cache_path)
