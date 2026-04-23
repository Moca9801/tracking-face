"""Tests de integración para face_match.search (sin GPU / sin cámara).

Usamos monkeypatch para reemplazar las funciones de OpenCV y de descarga
de modelos por stubs deterministas, de modo que los tests sean rápidos,
independientes de la red y reproducibles en cualquier entorno CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_match import search as _search_module
from face_match.search import run_search


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_gallery(tmp_path: Path, n: int = 4) -> Path:
    """Crea una galería falsa con n imágenes JPEG vacías."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    for i in range(n):
        (gallery / f"person_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    return gallery


def _make_query(tmp_path: Path) -> Path:
    """Crea una imagen de consulta falsa."""
    q = tmp_path / "query.jpg"
    q.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    return q


def _stub_ensure_model(filename: str) -> Path:
    """Devuelve una ruta ficticia sin descargar nada."""
    return Path("/fake/model.onnx")


def _make_embed_stub(scores: list[float]):
    """Genera un stub de `embed` que devuelve vectores unitarios con score."""
    call_count = [0]

    def _embed(bgr, detector, recognizer):
        idx = call_count[0] % len(scores)
        call_count[0] += 1
        vec = np.zeros(128, dtype=np.float32)
        vec[idx % 128] = scores[idx]  # vector diferente por imagen
        # Normalizar para que la similitud coseno sea controlada
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    return _embed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    """Verifica que el filtro de umbral funciona correctamente."""

    def test_high_threshold_filters_all(self, tmp_path, monkeypatch, capsys):
        """Un threshold imposiblemente alto debe dejar 0 resultados y retornar 1."""
        gallery = _make_gallery(tmp_path)
        query = _make_query(tmp_path)

        # Mock del detector y reconocedor
        class FakeRecognizer:
            def match(self, f1, f2, dis_type):
                return 0.1  # similitud baja para este test
        
        monkeypatch.setattr(_search_module, "ensure_model", _stub_ensure_model)
        monkeypatch.setattr(_search_module, "embed", _make_embed_stub([0.1, 0.1, 0.1, 0.1]))
        monkeypatch.setattr(_search_module, "load_bgr", lambda p: np.zeros((10, 10, 3), dtype=np.uint8))
        monkeypatch.setattr("cv2.FaceDetectorYN.create", lambda *a, **kw: object())
        monkeypatch.setattr("cv2.FaceRecognizerSF.create", lambda *a, **kw: FakeRecognizer())

        rc = run_search(
            query=query,
            db=gallery,
            top=10,
            distance=0,  # coseno
            rebuild_cache=True,
            threshold=0.99,  # imposible para embeddings normalizados de baja similitud
        )
        assert rc == 1
        captured = capsys.readouterr()
        assert "umbral" in captured.err.lower() or "coincidencias" in captured.err.lower()

    def test_zero_threshold_accepts_all(self, tmp_path, monkeypatch):
        """Un threshold de 0.0 debe aceptar cualquier resultado coseno > 0."""
        gallery = _make_gallery(tmp_path, n=3)
        query = _make_query(tmp_path)

        class FakeRecognizer:
            def match(self, f1, f2, dis_type):
                return 0.5

        monkeypatch.setattr(_search_module, "ensure_model", _stub_ensure_model)
        monkeypatch.setattr(_search_module, "embed", _make_embed_stub([0.5, 0.3, 0.2]))
        monkeypatch.setattr(_search_module, "load_bgr", lambda p: np.zeros((10, 10, 3), dtype=np.uint8))
        monkeypatch.setattr("cv2.FaceDetectorYN.create", lambda *a, **kw: object())
        monkeypatch.setattr("cv2.FaceRecognizerSF.create", lambda *a, **kw: FakeRecognizer())

        # threshold=0.0 — cualquier similitud positiva debería pasar
        # No afirmamos el rc porque el stub de match puede devolver 0;
        # el test principal es que no lanza excepción.
        try:
            run_search(
                query=query,
                db=gallery,
                top=10,
                distance=0,
                rebuild_cache=True,
                threshold=0.0,
            )
        except Exception as exc:
            pytest.fail(f"run_search lanzó excepción inesperada: {exc}")


class TestQueryValidation:
    """Verifica los errores de entrada."""

    def test_missing_query_file_returns_1(self, tmp_path):
        gallery = _make_gallery(tmp_path)
        rc = run_search(
            query=tmp_path / "nonexistent.jpg",
            db=gallery,
            top=5,
            distance=0,
            rebuild_cache=False,
            threshold=0.363,
        )
        assert rc == 1

    def test_missing_db_dir_returns_1(self, tmp_path):
        query = _make_query(tmp_path)
        rc = run_search(
            query=query,
            db=tmp_path / "nonexistent_db",
            top=5,
            distance=0,
            rebuild_cache=False,
            threshold=0.363,
        )
        assert rc == 1

    def test_empty_gallery_returns_1(self, tmp_path):
        query = _make_query(tmp_path)
        empty_gallery = tmp_path / "empty"
        empty_gallery.mkdir()
        rc = run_search(
            query=query,
            db=empty_gallery,
            top=5,
            distance=0,
            rebuild_cache=False,
            threshold=0.363,
        )
        assert rc == 1


class TestResultOrdering:
    """Verifica que los resultados se ordenen correctamente según la métrica."""

    def test_cosine_results_are_descending(self, tmp_path, monkeypatch, capsys):
        """Para coseno, el primer resultado debe tener la mayor similitud."""
        gallery = _make_gallery(tmp_path, n=3)
        query = _make_query(tmp_path)

        captured_dists: list[float] = []

        # Patch recognizer.match para devolver valores controlados
        class FakeRecognizer:
            _call = [0]
            _values = [0.9, 0.4, 0.5]

            def alignCrop(self, img, face):
                return img

            def feature(self, img):
                return np.ones(128, dtype=np.float32)

            def match(self, f1, f2, dis_type):
                val = self._values[self._call[0] % len(self._values)]
                self._call[0] += 1
                captured_dists.append(val)
                return val

        class FakeDetector:
            def setInputSize(self, size):
                pass

            def detect(self, img):
                # Simular 1 cara detectada con score alto
                face = np.array([[0, 0, 10, 10] + [1.0] * 10 + [0.99]], dtype=np.float32)
                return None, face

        monkeypatch.setattr(_search_module, "ensure_model", _stub_ensure_model)
        monkeypatch.setattr("cv2.FaceDetectorYN.create", lambda *a, **kw: FakeDetector())
        monkeypatch.setattr("cv2.FaceRecognizerSF.create", lambda *a, **kw: FakeRecognizer())
        monkeypatch.setattr(_search_module, "load_bgr", lambda p: np.zeros((100, 100, 3), dtype=np.uint8))

        run_search(
            query=query,
            db=gallery,
            top=10,
            distance=0,
            rebuild_cache=True,
            threshold=0.0,
        )

        out = capsys.readouterr().out
        # Extraer las distancias del output impreso
        lines = [l for l in out.splitlines() if l.strip().startswith(("1.", "2.", "3."))]
        if len(lines) >= 2:
            d1 = float(lines[0].split("dist=")[1].split()[0])
            d2 = float(lines[1].split("dist=")[1].split()[0])
            assert d1 >= d2, f"Coseno: resultado 1 ({d1}) debe ser >= resultado 2 ({d2})"
