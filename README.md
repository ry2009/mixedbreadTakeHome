# Mixedbread Multivector Store

I built this repository as a disk-backed multivector store tailored for late-interaction models (ColBERT, ColPali, etc.).
I focus on **cold-start latency** and **memory efficiency** while keeping the API minimal: `insert`, `delete`, and `search`.

## Highlights
- I rely on persistent storage on NVMe SSDs using a compact binary layout (`vectors.bin` + `index.pkl`).
- I use zero-copy reads via `os.pread` and NumPy buffer views; there is no need to memory-map the full index.
- I let scoring backends be selectable at runtime: [`maxsim-cpu`](https://github.com/mixedbread-ai/maxsim-cpu) for peak latency or a pure NumPy fallback when the native kernel is unavailable.
- I enable optional threadpool parallelism for NumPy scoring to accelerate CPU-heavy workloads when native kernels are unavailable.
- I ship CLI tooling for building synthetic datasets and running reproducible benchmarks.
- I manage fragmentation automatically: once at least 64 MiB of garbage accumulates and exceeds 35 % of on-disk bytes the store triggers compaction in-place, while manual compaction remains exposed via CLI/API.

## Project Layout
```
pyproject.toml
src/mixedbread_store/
  __init__.py          # Public API exposing insert/delete/search and the store class
  store.py             # DiskBackedMultiVectorStore implementation
  cli.py               # CLI entry-point (mvstore-benchmark)
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Populate the store with synthetic data
I mirrored the data generation script from the prompt and added batching plus repeatable seeds.

```bash
mvstore-benchmark --store-dir ./data build --count 100000 --batch-size 512 --reset
```

- `--reset` clears existing store files by removing `vectors.bin`/`index.pkl`.
- I generate vectors as `int8` with per-document sequence lengths sampled uniformly between 10 and 250 tokens.

### Run latency benchmarks
```bash
# Drop filesystem caches if you have root access (optional but recommended for cold start numbers)
# echo 3 | sudo tee /proc/sys/vm/drop_caches

mvstore-benchmark --store-dir ./data benchmark --searches 101 --candidates 1000
```

My benchmark script samples `candidates` document IDs per query, generates a `(32, 128)` float32 query, and reports:
- `Cold start latency` (first call, cache-cold)
- `Warm avg latency` across the remaining calls
- `Warm p95 latency`

### Backend selection
By default the store automatically uses `maxsim-cpu` when it’s importable. To force the NumPy fallback (for benchmarks or constrained environments), I set `MVSTORE_DISABLE_MAXSIM=1` or instantiate `DiskBackedMultiVectorStore(..., backend="numpy")`. I tune threadpool parallelism for the NumPy path via `max_workers` (defaults to roughly half the visible hardware threads; pass 1 to disable).

### Compact the store
After large batches of deletions, I run compaction to reclaim SSD space:

```bash
mvstore-benchmark --store-dir ./data compact
```

This command rewrites `vectors.bin` in place and persists the refreshed index for me.

#### Auto-compaction heuristics
- Compaction runs automatically when the store exceeds 64 MiB and at least 35 % of the file consists of tombstoned (deleted or superseded) records.
- Change the behaviour per tenant by instantiating `DiskBackedMultiVectorStore` with `auto_compact_ratio`/`auto_compact_min_bytes`, or disable auto-compaction by setting the ratio to `None`.

### Using the API directly
```python
import numpy as np
from mixedbread_store import insert, delete, search, compact

# Insert two documents
insert(
    [1, 2],
    [
        np.random.randint(-127, 128, size=(64, 128), dtype=np.int8),
        np.random.randint(-127, 128, size=(48, 128), dtype=np.int8),
    ],
)

# Run MaxSim search over a candidate set
query = np.random.randn(32, 128).astype(np.float32)
results = search([1, 2], query)
print(results)  # [(doc_id, score), ...] sorted by score descending

# Delete documents
delete([1])

# Reclaim storage if many deletes accumulated (optional—auto-compaction is on by default)
compact()
```

For multi-tenant deployments I instantiate a dedicated `DiskBackedMultiVectorStore` pointing to an isolated directory.

### End-to-end stage-2 inference example
```python
import numpy as np
from pathlib import Path
from mixedbread_store import DiskBackedMultiVectorStore

store_dir = Path("./demo_store")
store_dir.mkdir(exist_ok=True)

# 1. Insert documents returned by the first-stage retriever (token embeddings)
docs = {
    101: np.random.randint(-5, 6, size=(200, 128), dtype=np.int8),
    202: np.random.randint(-5, 6, size=(180, 128), dtype=np.int8),
    303: np.random.randint(-5, 6, size=(150, 128), dtype=np.int8),
    404: np.random.randint(-5, 6, size=(90, 128), dtype=np.int8),
}
with DiskBackedMultiVectorStore(store_dir, backend="numpy", auto_compact_ratio=None, max_workers=1) as store:
    store.delete(store.doc_ids())  # reset for demo
    store.insert(list(docs.keys()), list(docs.values()))

# 2. Prefiltered candidate IDs (from Stage 1)
candidates = [101, 202, 303, 404]

# 3. Stage-2 MaxSim scoring against a query embedding
query = np.random.randn(32, 128).astype(np.float32)
with DiskBackedMultiVectorStore(store_dir, backend="numpy", auto_compact_ratio=None, max_workers=1) as store:
    scores = store.search(candidates, query)

print("Stage-2 MaxSim scores:")
for rank, (doc_id, score) in enumerate(scores, start=1):
    print(f"{rank:>2}. doc_id={doc_id}, score={score:.2f}")

# 4. Cleanup
import shutil
shutil.rmtree(store_dir, ignore_errors=True)
```
This mirrors the intended production flow I target: Stage 1 narrows millions of documents to ~1 000 candidates, Stage 2 streams their token embeddings off NVMe, runs MaxSim against the query, and returns a ranked list while holding only the candidate set in memory.

## Benchmarks
I captured all timings on my local development machine (`Darwin 24.6.0, 8-core Intel`) using Python 3.11 and the CLI commands shown above.

| Dataset | Store size | Cold start | Warm avg (100 runs) | Warm p95 |
|---------|------------|------------|---------------------|----------|
| 100k docs, seq_len∈[10,250], dim=128 (macOS dev box) | 1.6 GB (`du -h bench_store`) | 313 ms | 313 ms | 326 ms |
| 100k docs, seq_len∈[10,250], dim=128 (Docker Desktop linux/overlayfs) | 1.6 GB | 860 ms | 624 ms | 866 ms |

Notes:
- I executed the cold-start test immediately after store construction (macOS lacks `drop_caches`).
- Each search touches 1,000 candidate documents and performs NumPy MaxSim scoring against a `(32, 128)` query.
- Warm calls stay stable thanks to the OS page cache and contiguous on-disk layout.
- Docker Desktop numbers are conservative due to overlayfs and virtualised storage; I expect materially lower latency on the target `c4-standard-8-lssd` once I run benchmarks there after `echo 3 > /proc/sys/vm/drop_caches`.
- TODO: capture official `c4-standard-8-lssd` metrics when the cloud instance is available.

## Design Notes
- **Data layout**: I store each record as `(doc_id:uint64, seq_len:uint32, dim:uint32, int8 vectors...)`. Append-only writes plus a compact pickled index keep persistence simple and S3-friendly for me.
- **Reads**: I use `os.pread` to stream records without disturbing the file pointer, allowing concurrent search threads. NumPy views avoid copies when reshaping the int8 payload.
- **MaxSim**: I convert queries to float32 once per call; vectors are row-normalised before BLAS multiplication so NumPy and `maxsim-cpu` return identical scores.
- **Deletes / updates**: Deletions and reinserts mark old segments as garbage. I track tombstone bytes and auto-compact when fragmentation crosses configurable thresholds.
- **Parallel fallback**: When native kernels are unavailable I fan out the NumPy path across a threadpool (defaults to half the hardware threads) so the BLAS-backed matmuls can run concurrently.

### Inference pipeline at a glance
1. **Prefiltered candidates** arrive from the first-stage retriever as up to 1 000 document IDs.
2. **Vector streaming** pulls each document’s token embeddings from `vectors.bin` using `os.pread`, keeping peak memory proportional to the candidate set.
3. **MaxSim scoring** runs either via `maxsim-cpu` (preferred) or the multi-threaded NumPy fallback. Each query token finds its best matching document token; scores are summed to produce the final relevance signal.
4. **Ranking** sorts the candidate list in-memory and returns `(doc_id, score)` pairs ready for downstream fusion with metadata or business logic.

## Possible Extensions
1. I could parallelize I/O and scoring across multiple worker threads when `maxsim-cpu` is present.
2. I could expose fragmentation metrics via Prometheus to monitor compaction frequency across tenants.
3. I could extend the index to store auxiliary metadata (e.g., tenant IDs, document lengths) for smarter candidate pruning prior to MaxSim scoring.

## Testing
- I run `python -m compileall src` to ensure the package imports cleanly.
- I use the `mvstore-benchmark` CLI to validate end-to-end insertion, search, compaction, and latency benchmarks.
- I performed a Docker Desktop quick check (Linux VM, overlayfs storage):
  ```bash
  docker run --rm -v "$PWD":/workspace -w /workspace python:3.11 \
    bash -lc 'apt-get update >/tmp/apt.log && \
              apt-get install -y libopenblas-dev >/tmp/apt.log && \
              python -m pip install -e . >/tmp/pip.log && \
              python -m pip install git+https://github.com/mixedbread-ai/maxsim-cpu >/tmp/pip2.log && \
              mvstore-benchmark --store-dir ./bench_store build --count 100000 --batch-size 512 --reset && \
              mvstore-benchmark --store-dir ./bench_store benchmark --searches 101 --candidates 1000'
  ```
  (This run outputs the 860 ms / 624 ms / 866 ms figures shown above.)

## Optional acceleration
The store automatically uses `maxsim-cpu` if the Python bindings are importable. I recommend installing it on Linux with:

```bash
python -m pip install git+https://github.com/mixedbread-ai/maxsim-cpu
```

Quick verification (this should print identical score lists twice for me):

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from mixedbread_store.store import DiskBackedMultiVectorStore, maxsim_cpu
from mixedbread_store import store as store_module
from pathlib import Path

assert maxsim_cpu is not None, "maxsim-cpu import failed"
with DiskBackedMultiVectorStore(Path('./verify_store'), auto_compact_ratio=None) as store:
    store.delete(store.doc_ids())
    docs = [np.random.randint(-5, 6, size=(32, 128), dtype=np.int8) for _ in range(5)]
    store.insert(range(5), docs)
    query = np.random.randn(32, 128).astype(np.float32)
    accel = store.search(range(5), query)
    store_module.maxsim_cpu = None
    try:
        pure = store.search(range(5), query)
    finally:
        store_module.maxsim_cpu = maxsim_cpu
    print(accel)
    print(pure)
PY
rm -rf verify_store
```

On unsupported platforms the code transparently falls back to the NumPy implementation.
