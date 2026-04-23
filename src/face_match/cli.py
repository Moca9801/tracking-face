# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from face_match.search import run_search


def main() -> int:
    default_db = Path.cwd() / "database"
    parser = argparse.ArgumentParser(
        description="Encuentra fotos con rostros similares a una imagen de consulta."
    )
    parser.add_argument("query", type=Path, help="Imagen con el rostro a buscar.")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Carpeta con la galería (recursivo). Por defecto: {default_db} (carpeta actual).",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=10, help="Máximo de resultados a listar."
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Distancia entre vectores (por defecto cosine / FR_COSINE).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Umbral mínimo de similitud. Recomendado: cosine=0.363, l2=1.128.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignora caché y vuelve a extraer descriptores de toda la base.",
    )
    args = parser.parse_args()
    dist = 0 if args.metric == "cosine" else 1

    threshold = args.threshold
    if threshold is None:
        threshold = 0.363 if dist == 0 else 1.128
    else:
        # Validar rango según la métrica seleccionada
        if dist == 0 and not (0.0 <= threshold <= 1.0):
            parser.error(
                f"--threshold {threshold!r} fuera de rango para coseno. "
                "Debe estar entre 0.0 y 1.0 (ej. 0.363)."
            )
        elif dist == 1 and threshold <= 0.0:
            parser.error(
                f"--threshold {threshold!r} inválido para L2. "
                "Debe ser un número positivo (ej. 1.128)."
            )

    return run_search(
        query=args.query,
        db=args.db,
        top=max(1, args.top),
        distance=dist,
        rebuild_cache=args.rebuild,
        threshold=threshold,
    )
