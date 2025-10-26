from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from .store import DiskBackedMultiVectorStore, DEFAULT_STORE_DIR


def _build_index(
    store: DiskBackedMultiVectorStore,
    count: int,
    seed: int,
    min_seq_len: int,
    max_seq_len: int,
    embed_dim: int,
    batch_size: int,
) -> None:
    rng = np.random.default_rng(seed)
    doc_ids: List[int] = []
    vectors: List[np.ndarray] = []

    for doc_id in tqdm(range(count), desc="Building index"):
        seq_len = int(rng.integers(min_seq_len, max_seq_len + 1))
        vec = rng.integers(
            low=-127,
            high=128,
            size=(seq_len, embed_dim),
            dtype=np.int8,
        )
        doc_ids.append(doc_id)
        vectors.append(vec)
        if len(doc_ids) >= batch_size:
            store.insert(doc_ids, vectors)
            doc_ids.clear()
            vectors.clear()

    if doc_ids:
        store.insert(doc_ids, vectors)


def _run_benchmark(
    store: DiskBackedMultiVectorStore,
    searches: int,
    candidates: int,
    query_tokens: int,
    embed_dim: int,
    seed: int,
) -> None:
    doc_ids = store.doc_ids()
    if len(doc_ids) < candidates:
        raise ValueError(
            f"Store contains {len(doc_ids)} documents, cannot sample {candidates} candidates"
        )

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    timings: List[float] = []

    for i in range(searches):
        sampled_ids = rng.sample(doc_ids, candidates)
        query = np_rng.normal(loc=0.0, scale=1.0, size=(query_tokens, embed_dim)).astype(
            np.float32
        )

        start = time.perf_counter()
        results = store.search(sampled_ids, query)
        duration = time.perf_counter() - start
        timings.append(duration)
        if not results:
            raise RuntimeError("Search returned no results; verify the index is populated")

    cold = timings[0]
    warm = timings[1:]
    warm_avg = sum(warm) / max(len(warm), 1)
    warm_p95 = np.percentile(warm, 95) if warm else warm_avg

    print(f"Cold start latency: {cold * 1000:.2f} ms")
    if warm:
        print(f"Warm avg latency ({len(warm)} runs): {warm_avg * 1000:.2f} ms")
        print(f"Warm p95 latency: {warm_p95 * 1000:.2f} ms")


def _reset_store(directory: Path) -> None:
    data_path = directory / "vectors.bin"
    index_path = directory / "index.pkl"
    for path in (data_path, index_path):
        if path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disk-backed multivector store utilities")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help=f"Directory containing store files (default: {DEFAULT_STORE_DIR})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Populate the store with synthetic data")
    build_cmd.add_argument("--count", type=int, default=100_000, help="Number of documents")
    build_cmd.add_argument("--seed", type=int, default=0, help="Random seed")
    build_cmd.add_argument("--min-seq-len", type=int, default=10)
    build_cmd.add_argument("--max-seq-len", type=int, default=250)
    build_cmd.add_argument("--embed-dim", type=int, default=128)
    build_cmd.add_argument("--batch-size", type=int, default=512)
    build_cmd.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing store files before building",
    )

    bench_cmd = subparsers.add_parser("benchmark", help="Run search latency benchmark")
    bench_cmd.add_argument("--searches", type=int, default=101)
    bench_cmd.add_argument("--candidates", type=int, default=1000)
    bench_cmd.add_argument("--query-tokens", type=int, default=32)
    bench_cmd.add_argument("--embed-dim", type=int, default=128)
    bench_cmd.add_argument("--seed", type=int, default=1234)

    subparsers.add_parser("compact", help="Rewrite storage to drop deleted records")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    store_dir: Path = args.store_dir
    store_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "build":
        if args.reset:
            _reset_store(store_dir)
        store = DiskBackedMultiVectorStore(store_dir)
        try:
            _build_index(
                store=store,
                count=args.count,
                seed=args.seed,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
                embed_dim=args.embed_dim,
                batch_size=args.batch_size,
            )
        finally:
            store.close()
    elif args.command == "benchmark":
        store = DiskBackedMultiVectorStore(store_dir)
        try:
            _run_benchmark(
                store=store,
                searches=args.searches,
                candidates=args.candidates,
                query_tokens=args.query_tokens,
                embed_dim=args.embed_dim,
                seed=args.seed,
            )
        finally:
            store.close()
    elif args.command == "compact":
        store = DiskBackedMultiVectorStore(store_dir)
        try:
            before = store._data_path.stat().st_size if store._data_path.exists() else 0
            store.compact()
            after = store._data_path.stat().st_size if store._data_path.exists() else 0
        finally:
            store.close()
        print(f"Compacted store: {before} -> {after} bytes")
    else:
        parser.error(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
