from __future__ import annotations

import numpy as np
import pytest

from mixedbread_store.store import DiskBackedMultiVectorStore


def test_search_returns_expected_score(tmp_path) -> None:
    doc = np.ones((32, 128), dtype=np.int8)
    query = np.ones((32, 128), dtype=np.float32)

    with DiskBackedMultiVectorStore(tmp_path, auto_compact_ratio=None) as store:
        store.insert([101], [doc])
        results = store.search([101], query)

    assert len(results) == 1
    doc_id, score = results[0]
    assert doc_id == 101
    assert score == pytest.approx(32.0, rel=1e-5)


def test_search_raises_on_dimension_mismatch(tmp_path) -> None:
    doc = np.ones((16, 64), dtype=np.int8)
    query = np.ones((32, 128), dtype=np.float32)

    with DiskBackedMultiVectorStore(tmp_path, auto_compact_ratio=None) as store:
        store.insert([202], [doc])
        with pytest.raises(ValueError):
            store.search([202], query)

