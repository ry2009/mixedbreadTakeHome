from __future__ import annotations

import os
import pickle
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import maxsim_cpu  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    maxsim_cpu = None


_TRUE_VALUES = {"1", "true", "yes", "on"}

_disable_maxsim_env = os.environ.get("MVSTORE_DISABLE_MAXSIM", "")
_USE_MAXSIM_BY_DEFAULT = maxsim_cpu is not None and (
    _disable_maxsim_env.strip().lower() not in _TRUE_VALUES
)

DEFAULT_STORE_DIR = Path(os.environ.get("MVSTORE_DIR", "./mvstore_data")).resolve()
DATA_FILENAME = "vectors.bin"
INDEX_FILENAME = "index.pkl"

HEADER_STRUCT = struct.Struct("<QII")  # doc_id, seq_len, embed_dim
INDEX_VERSION = 1
AUTO_COMPACT_RATIO_DEFAULT = 0.35
AUTO_COMPACT_MIN_BYTES_DEFAULT = 64 * 1024 * 1024  # 64 MiB


@dataclass(frozen=True)
class IndexEntry:
    offset: int  # Offset where the record header begins
    seq_len: int
    embed_dim: int


class DiskBackedMultiVectorStore:
    """
    Persistent disk-backed multivector store optimized for MaxSim ranking.

    Storage layout:
      - `vectors.bin` stores a sequence of records. Each record packs
        (doc_id, seq_len, embed_dim) followed by seq_len * embed_dim bytes (int8).
      - `index.pkl` stores a versioned dictionary {doc_id: IndexEntry}.
    """

    def __init__(
        self,
        directory: Path | str = DEFAULT_STORE_DIR,
        *,
        auto_compact_ratio: float | None = AUTO_COMPACT_RATIO_DEFAULT,
        auto_compact_min_bytes: int = AUTO_COMPACT_MIN_BYTES_DEFAULT,
        backend: str | None = None,
        max_workers: int | None = None,
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._data_path = self._dir / DATA_FILENAME
        self._index_path = self._dir / INDEX_FILENAME
        self._lock = threading.RLock()
        self._auto_compact_ratio = auto_compact_ratio
        self._auto_compact_min_bytes = auto_compact_min_bytes
        self._garbage_bytes = 0
        self._suppress_auto_compact = False

        backend_choice = backend.lower() if backend else None
        if backend_choice not in {None, "maxsim", "numpy"}:
            raise ValueError("backend must be 'maxsim', 'numpy', or None")
        if backend_choice == "maxsim" and maxsim_cpu is None:
            backend_choice = "numpy"
        self._use_maxsim = (
            _USE_MAXSIM_BY_DEFAULT
            if backend_choice is None
            else (backend_choice == "maxsim")
        )
        if backend_choice == "numpy":
            self._use_maxsim = False
        if max_workers is None:
            hw_threads = os.cpu_count() or 1
            max_workers = max(1, min(32, hw_threads // 2)) if hw_threads > 1 else 1
        self._executor = (
            ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
        )

        # Ensure data file exists, then open without O_APPEND so offsets are reliable.
        self._data_path.touch(exist_ok=True)
        self._data_file = open(self._data_path, "r+b", buffering=0)
        self._fd = self._data_file.fileno()

        self._index: Dict[int, IndexEntry] = {}
        self._load_index()

    def close(self) -> None:
        with self._lock:
            self._data_file.close()
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def __enter__(self) -> "DiskBackedMultiVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not self._index_path.exists():
            self._index = {}
            return
        with self._index_path.open("rb") as fh:
            payload = pickle.load(fh)
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise RuntimeError(
                f"Incompatible index version {version}; expected {INDEX_VERSION}"
            )
        entries = payload.get("entries", {})
        self._index = {
            int(doc_id): IndexEntry(*entry) for doc_id, entry in entries.items()
        }
        self._garbage_bytes = int(payload.get("garbage_bytes", 0))

    def _write_all(self, data: memoryview | bytes) -> None:
        self._write_all_fd(self._fd, data)

    @staticmethod
    def _write_all_fd(fd: int, data: memoryview | bytes) -> None:
        view = memoryview(data)
        total = len(view)
        written = 0
        while written < total:
            n = os.write(fd, view[written:])
            if n == 0:
                raise IOError("os.write returned 0 bytes written")
            written += n

    def _persist_index(self) -> None:
        payload = {
            "version": INDEX_VERSION,
            "garbage_bytes": self._garbage_bytes,
            "entries": {
                doc_id: (entry.offset, entry.seq_len, entry.embed_dim)
                for doc_id, entry in self._index.items()
            },
        }
        tmp_path = self._index_path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(self._index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, doc_ids: Sequence[int], vectors: Sequence[np.ndarray]) -> None:
        if len(doc_ids) != len(vectors):
            raise ValueError("doc_ids and vectors must have identical lengths")
        if not doc_ids:
            return
        with self._lock:
            for doc_id, vector in zip(doc_ids, vectors):
                if vector.ndim != 2:
                    raise ValueError("Each vector must be a 2D array [seq_len, embed_dim]")
                if vector.dtype != np.int8 or not vector.flags.c_contiguous:
                    vector = np.ascontiguousarray(vector, dtype=np.int8)
                seq_len, embed_dim = vector.shape
                if embed_dim != 128:
                    # Embed dim is technically flexible, but MaxSim kernel currently assumes 128
                    # so we keep metadata for correctness.
                    pass

                doc_id = int(doc_id)
                existing = self._index.get(doc_id)
                if existing is not None:
                    self._garbage_bytes += self._record_size(existing)
                record_offset = os.lseek(self._fd, 0, os.SEEK_END)
                header = HEADER_STRUCT.pack(int(doc_id), int(seq_len), int(embed_dim))
                self._write_all(header)
                self._write_all(vector.tobytes(order="C"))
                self._index[doc_id] = IndexEntry(
                    offset=record_offset,
                    seq_len=int(seq_len),
                    embed_dim=int(embed_dim),
                )
            self._data_file.flush()
            os.fsync(self._fd)
            self._persist_index()
            self._maybe_auto_compact()

    def delete(self, doc_ids: Iterable[int]) -> None:
        changed = False
        with self._lock:
            for doc_id in doc_ids:
                doc_id = int(doc_id)
                entry = self._index.pop(doc_id, None)
                if entry is not None:
                    self._garbage_bytes += self._record_size(entry)
                    changed = True
            if changed:
                self._persist_index()
                self._maybe_auto_compact()

    def compact(self) -> None:
        with self._lock:
            self._perform_compact(force=True)

    def doc_ids(self) -> List[int]:
        with self._lock:
            return list(self._index.keys())

    @staticmethod
    def _record_size(entry: IndexEntry) -> int:
        return HEADER_STRUCT.size + entry.seq_len * entry.embed_dim

    def _perform_compact(self, *, force: bool) -> None:
        if self._suppress_auto_compact and not force:
            return
        if self._suppress_auto_compact and force:
            return

        self._suppress_auto_compact = True
        try:
            if not self._index:
                self._data_file.close()
                self._data_file = open(self._data_path, "w+b", buffering=0)
                self._fd = self._data_file.fileno()
                self._garbage_bytes = 0
                self._persist_index()
                return

            self._data_file.flush()
            os.fsync(self._fd)

            temp_path = self._data_path.with_suffix(".compacting")
            new_index: Dict[int, IndexEntry] = {}
            try:
                with open(temp_path, "w+b", buffering=0) as new_file:
                    new_fd = new_file.fileno()
                    offset = 0
                    for doc_id in sorted(self._index.keys()):
                        entry = self._index[doc_id]
                        record_size = self._record_size(entry)
                        raw = os.pread(self._fd, record_size, entry.offset)
                        if len(raw) != record_size:
                            continue
                        stored_doc_id, _, _ = HEADER_STRUCT.unpack_from(raw)
                        if stored_doc_id != doc_id:
                            continue
                        self._write_all_fd(new_fd, raw)
                        new_index[doc_id] = IndexEntry(
                            offset=offset,
                            seq_len=entry.seq_len,
                            embed_dim=entry.embed_dim,
                        )
                        offset += record_size
                    new_file.flush()
                    os.fsync(new_fd)

                self._data_file.close()
                os.replace(temp_path, self._data_path)
                self._data_file = open(self._data_path, "r+b", buffering=0)
                self._fd = self._data_file.fileno()
                self._index = new_index
                self._garbage_bytes = 0
                self._persist_index()
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        finally:
            self._suppress_auto_compact = False

    def _maybe_auto_compact(self) -> None:
        if self._suppress_auto_compact:
            return
        if self._auto_compact_ratio is None:
            return
        file_size = self._data_path.stat().st_size if self._data_path.exists() else 0
        if file_size <= self._auto_compact_min_bytes:
            return
        if file_size == 0:
            return
        ratio = self._garbage_bytes / file_size
        if ratio < self._auto_compact_ratio:
            return
        self._perform_compact(force=False)

    def search(
        self,
        doc_ids: Sequence[int],
        query_vector: np.ndarray,
    ) -> List[Tuple[int, float]]:
        if query_vector.ndim != 2:
            raise ValueError("query_vector must be 2D [num_tokens, embed_dim]")
        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        query_norm = _normalize_rows(query)
        query_dim = query_norm.shape[1]

        docs: List[Tuple[int, np.ndarray]] = []
        for doc_id in doc_ids:
            entry = self._index.get(int(doc_id))
            if entry is None:
                continue
            record_size = self._record_size(entry)
            raw = os.pread(self._fd, record_size, entry.offset)
            if len(raw) != record_size:
                # Corrupted entry; skip gracefully
                continue
            stored_doc_id, seq_len, embed_dim = HEADER_STRUCT.unpack_from(raw)
            if stored_doc_id != int(doc_id):
                # Index might be stale; skip for safety.
                continue
            if embed_dim != query_dim:
                raise ValueError(
                    "Query embedding dimension {query_dim} does not match document "
                    "{doc_id} embedding dimension {embed_dim}".format(
                        query_dim=query_dim, doc_id=int(doc_id), embed_dim=embed_dim
                    )
                )
            doc_buffer = memoryview(raw)[HEADER_STRUCT.size :]
            doc_int8 = np.frombuffer(doc_buffer, dtype=np.int8).reshape(seq_len, embed_dim)
            doc = np.ascontiguousarray(doc_int8, dtype=np.float32)
            docs.append((int(doc_id), _normalize_rows(doc)))

        if not docs:
            return []

        doc_ids_order = [doc_id for (doc_id, _) in docs]
        normalized_docs = [doc for (_, doc) in docs]

        scores: List[Tuple[int, float]]
        if self._use_maxsim and maxsim_cpu is not None:
            raw_scores = maxsim_cpu.maxsim_scores_variable(query_norm, normalized_docs)
            scores = [(doc_id, float(score)) for doc_id, score in zip(doc_ids_order, raw_scores)]
        else:
            if self._executor is not None and len(docs) > 1:
                def task(item: Tuple[int, np.ndarray]) -> Tuple[int, float]:
                    doc_id, doc = item
                    return doc_id, float(self._maxsim_numpy(query_norm, doc))

                scores = list(self._executor.map(task, docs))
            else:
                scores = [
                    (doc_id, float(self._maxsim_numpy(query_norm, doc)))
                    for doc_id, doc in docs
                ]

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    @staticmethod
    def _maxsim_numpy(query: np.ndarray, doc: np.ndarray) -> float:
        sims = query @ doc.T  # (n_query_tokens, seq_len)
        max_per_query = np.max(sims, axis=1)
        return float(np.sum(max_per_query, dtype=np.float32))


# ----------------------------------------------------------------------
# Module-level helper bound to a default store location
# ----------------------------------------------------------------------
_DEFAULT_STORE = DiskBackedMultiVectorStore()


def insert(doc_ids: Sequence[int], vectors: Sequence[np.ndarray]) -> None:
    _DEFAULT_STORE.insert(doc_ids, vectors)


def delete(doc_ids: Iterable[int]) -> None:
    _DEFAULT_STORE.delete(doc_ids)


def search(doc_ids: Sequence[int], query_vector: np.ndarray) -> List[Tuple[int, float]]:
    return _DEFAULT_STORE.search(doc_ids, query_vector)


def compact() -> None:
    _DEFAULT_STORE.compact()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return mat / norms
