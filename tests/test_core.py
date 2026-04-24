from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_match.core import list_image_paths, load_cache, pick_best_face, save_cache


def test_pick_best_face_empty() -> None:
    assert pick_best_face(None) is None
    assert pick_best_face(np.empty((0, 15))) is None


def test_pick_best_face_scores() -> None:
    # (x, y, w, h, ... 9 keypoint cols ..., score)
    a = np.array([[0, 0, 1, 1] + [0.0] * 10 + [0.1]], dtype=np.float32)
    b = np.array([[0, 0, 1, 1] + [0.0] * 10 + [0.9]], dtype=np.float32)
    faces = np.vstack([a, b])
    best = pick_best_face(faces)
    assert best is not None
    assert float(best[14]) == pytest.approx(0.9)


def test_list_image_paths_filters(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").write_bytes(b"\x89")
    out = list_image_paths(tmp_path)
    names = {p.name for p in out}
    assert names == {"a.jpg", "c.png"}


def test_cache_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    data = {"k": (1.0, np.array([1, 2, 3]))}
    save_cache(p, data)
    loaded = load_cache(p)
    assert len(loaded) == 1
    t, arr = loaded["k"]
    assert t == 1.0
    np.testing.assert_array_equal(arr, [1, 2, 3])


def test_cache_missing_returns_empty() -> None:
    assert load_cache(Path("/nonexistent/path/cache.json")) == {}


def test_list_image_paths_nested(tmp_path: Path) -> None:
    d = tmp_path / "n" / "d"
    d.mkdir(parents=True)
    (d / "f.jpeg").write_bytes(b"x")
    assert len(list_image_paths(tmp_path)) == 1
