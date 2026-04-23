# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import cv2
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

    q_feat = embed(q_img, detector, recognizer)
    if q_feat is None:
        print(
            "No se detectó ningún rostro en la imagen de consulta. "
            "Prueba otra foto, frontal y bien iluminada.",
            file=sys.stderr,
        )
        return 1

    dis_type = (
        cv2.FaceRecognizerSF_FR_COSINE
        if distance == 0
        else cv2.FaceRecognizerSF_FR_NORM_L2
    )
    metric = "coseno" if dis_type == cv2.FaceRecognizerSF_FR_COSINE else "L2"

    cache_path = db / CACHE_NAME
    cache: dict = {} if rebuild_cache else load_cache(cache_path)
    results: list[tuple[float, Path, str]] = []
    need_save = rebuild_cache
    q_feat_f = q_feat

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
        d = float(recognizer.match(q_feat_f, feat, dis_type))
        results.append((d, imp, metric))

    if need_save and cache:
        try:
            save_cache(cache_path, cache)
        except OSError as e:
            print(f"Aviso: no se pudo guardar caché {cache_path}: {e}", file=sys.stderr)

    results.sort(key=lambda x: x[0], reverse=(distance == 0))
    k = min(top, len(results))

    print()
    print(f"Consulta: {query}")
    print(f"Base: {db.resolve()}  ({len(images)} imágenes, {len(results)} con rostro)")
    desc = (
        "valores mayores = más parecido"
        if distance == 0
        else "valores menores = más parecido"
    )
    print(f"Métrica: {metric} ({desc})")
    print()
    for i in range(k):
        d, p, m = results[i]
        print(f"{i + 1}.  dist={d:.4f}  {m}  {p.resolve()}")

    if not results:
        print("No quedó ninguna imagen con rostro para comparar.", file=sys.stderr)
        return 1
    return 0
