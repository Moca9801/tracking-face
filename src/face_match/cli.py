# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from face_match import __version__
from face_match.search import run_search


def main() -> int:
    default_db = Path.cwd() / "database"
    parser = argparse.ArgumentParser(
        description="Find photos with faces similar to a query image."
    )
    parser.add_argument("query", type=Path, help="Path to the query image")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Path to the gallery folder. Default: {default_db}",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=5, help="Number of results to show"
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Similarity metric",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold. Recommended: cosine=0.363, l2=1.128.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the entire cache",
    )
    parser.add_argument(
        "--version", action="version", version=f"face-match {__version__}"
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
                f"--threshold {threshold!r} out of range for cosine. "
                "Must be between 0.0 and 1.0 (e.g., 0.363)."
            )
        elif dist == 1 and threshold <= 0.0:
            parser.error(
                f"--threshold {threshold!r} invalid for L2. "
                "Must be a positive number (e.g., 1.128)."
            )

    return run_search(
        query=args.query,
        db=args.db,
        top=max(1, args.top),
        distance=dist,
        rebuild_cache=args.rebuild,
        threshold=threshold,
        device=args.device,
    )
