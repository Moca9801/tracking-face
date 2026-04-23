# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import faiss
import numpy as np

from face_match.core import (
    CACHE_NAME,
    IMAGE_EXTENSIONS,
    SFACE_NAME,
    YUNET_NAME,
    embed,
    ensure_model,
    list_image_paths,
    load_bgr,
    load_cache,
    save_cache,
)


def run_search(
    query: Path,
    db: Path,
    top: int,
    distance: int,
    rebuild_cache: bool,
    threshold: float,
    device: str = "cpu",
) -> int:
    if not query.is_file():
        print(f"No existe la imagen de consulta: {query}", file=sys.stderr)
        return 1
    if not db.is_dir():
        print(f"No existe la carpeta de la base de datos: {db}", file=sys.stderr)
        return 1

    images = list_image_paths(db)
    if not images:
        print(
            f"La carpeta {db} no contiene imágenes ({', '.join(sorted(IMAGE_EXTENSIONS))}).",
            file=sys.stderr,
        )
        return 1

    det_path = ensure_model(YUNET_NAME)
    rec_path = ensure_model(SFACE_NAME)

    q_img = load_bgr(query)
    if q_img is None:
        print(f"No se pudo leer la imagen: {query}", file=sys.stderr)
        return 1
    h0, w0 = q_img.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(det_path), "", (w0, h0), 0.9, 0.3, 5000, 0, 0)
    recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    if device == "gpu":
        detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    q_feat = embed(q_img, detector, recognizer)
    if q_feat is None:
        print(
            "No se detectó ningún rostro en la imagen de consulta. "
            "Prueba otra foto, frontal y bien iluminada.",
            file=sys.stderr,
        )
        return 1

    cache_path = db / CACHE_NAME
    cache: dict = {} if rebuild_cache else load_cache(cache_path)
    
    # Recolección de embeddings
    all_feats = []
    all_paths = []
    need_save = rebuild_cache

    for imp in images:
        if imp.resolve() == query.resolve():
            continue
        if imp.suffix.lower() == ".onnx":
            continue
        
        mtime = imp.stat().st_mtime
        key = str(imp.resolve())
        entry = cache.get(key)
        feat: np.ndarray | None = None
        
        if (
            not rebuild_cache
            and isinstance(entry, tuple)
            and len(entry) == 2
            and entry[0] == mtime
        ):
            feat = np.asarray(entry[1])
        else:
            bgr = load_bgr(imp)
            if bgr is None:
                continue
            feat = embed(bgr, detector, recognizer)
            if feat is None:
                print(f"Sin rostro (omitida): {imp.name}", file=sys.stderr)
                continue
            cache[key] = (mtime, feat)
            need_save = True
        
        all_feats.append(feat.flatten())
        all_paths.append(imp)

    if need_save and cache:
        try:
            save_cache(cache_path, cache)
        except OSError as e:
            print(f"Aviso: no se pudo guardar caché {cache_path}: {e}", file=sys.stderr)

    if not all_feats:
        print("No se encontraron rostros en la galería para comparar.", file=sys.stderr)
        return 1

    # Preparar datos para FAISS
    xb = np.array(all_feats).astype("float32")
    xq = q_feat.reshape(1, -1).astype("float32")
    
    # FAISS: Normalizar si es coseno (Inner Product en vectores unidad = Coseno)
    if distance == 0:
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        index = faiss.IndexFlatIP(xb.shape[1])
        metric_name = "coseno"
        desc = "valores mayores = más parecido"
    else:
        index = faiss.IndexFlatL2(xb.shape[1])
        metric_name = "L2"
        desc = "valores menores = más parecido"

    index.add(xb)
    
    # Mover índice a GPU si se solicita
    used_device = "CPU"
    if device == "gpu":
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            used_device = "GPU"
        except Exception:
            print("Aviso: FAISS GPU no disponible. Usando CPU...", file=sys.stderr)
            used_device = "CPU"
    
    # Búsqueda vectorial
    D, I = index.search(xq, len(all_paths))
    
    # Filtrado por umbral y preparación de resultados
    results = []
    for dist_val, idx in zip(D[0], I[0]):
        if idx == -1: continue
        
        d = float(dist_val)
        # SFace match devuelve valores específicos, FAISS IP devuelve el producto punto.
        # Para vectores normalizados, son idénticos.
        
        if distance == 0: # Coseno
            if d < threshold: continue
        else: # L2
            if d > threshold: continue
            
        results.append((d, all_paths[idx], metric_name))

    # Ordenar (Coseno descendente, L2 ascendente)
    results.sort(key=lambda x: x[0], reverse=(distance == 0))
    k = min(top, len(results))

    print()
    print(f"Consulta: {query}")
    print(
        f"Base: {db.resolve()}  "
        f"({len(images)} imágenes escaneadas, "
        f"{len(all_paths)} con rostro detectado, "
        f"{len(results)} supera umbral {threshold:.3f})"
    )
    print(f"Métrica: {metric_name} ({desc}) [Motor: FAISS {used_device}]")
    print()

    if not results:
        print(
            f"Sin coincidencias. Ninguna imagen superó el umbral de similitud ({threshold:.3f}).",
            file=sys.stderr,
        )
        return 1

    for i in range(k):
        d, p, m = results[i]
        print(f"{i + 1}.  dist={d:.4f}  {m}  {p.resolve()}")

    return 0


