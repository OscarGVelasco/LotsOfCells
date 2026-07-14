"""Parallel permutation execution helper.

Every permutation-based test in lotsofcells (``lots_of_cells``,
``entropy_score``, ``gene_source_score``) routes through the single
:func:`run_permutations` helper here.  That gives us:

- one place to change the backend if we ever move off joblib,
- one place to control the "chunk vs per-permutation" trade-off,
- reproducible results across ``n_cores`` values via
  :class:`numpy.random.SeedSequence`.

Design (matches the R original's ``BiocParallel::bplapply`` pattern):

- Each worker gets a CHUNK of permutations, not a single permutation.
  Per-permutation dispatch would spend more time on IPC than on numpy.
- Chunks are same-sized (last one may be one shorter). Each chunk gets an
  independent, spawned :class:`numpy.random.SeedSequence` so that for a
  fixed ``n_cores`` and ``seed``, results are bit-for-bit reproducible
  across runs. Note that changing ``n_cores`` changes the number of
  spawned children and thus the exact random draws — results across
  different ``n_cores`` values are statistically equivalent but not
  numerically identical. This matches the convention in
  scikit-learn / scipy / joblib.
- If ``n_permutations`` is too small to keep the workers busy, we silently
  fall back to serial rather than dispatch a tiny job.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# Minimum permutations per worker before we bother going parallel.
# Below this, dispatch overhead dominates and serial is faster.
_MIN_PER_WORKER = 20


ChunkFn = Callable[[int, np.random.SeedSequence], np.ndarray]
"""A worker function.

``chunk_fn(chunk_size, seed_seq) -> np.ndarray``: run ``chunk_size``
permutations serially, seeding all randomness from ``seed_seq``, and
return a stacked ``(chunk_size, ...)`` array. The trailing dimensions
must be constant across chunks so results can be concatenated.
"""


def _resolve_workers(n_cores: Optional[int]) -> int:
    """Turn a user-facing ``n_cores`` value into an actual worker count.

    - ``None`` or ``1`` → 1 (serial).
    - ``-1`` → all available cores.
    - any positive int → that many workers.
    """
    if n_cores is None or n_cores == 1:
        return 1
    if n_cores == -1:
        try:
            from joblib import cpu_count
            return max(1, int(cpu_count()))
        except Exception:  # noqa: BLE001
            import os
            return max(1, os.cpu_count() or 1)
    return max(1, int(n_cores))


def run_permutations(
    chunk_fn: ChunkFn,
    n_permutations: int,
    n_cores: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Run ``n_permutations`` independent draws, chunked across workers.

    Parameters
    ----------
    chunk_fn
        See :data:`ChunkFn`. Called once per worker (or once total if
        serial), given a chunk size and a seeded ``SeedSequence``.
    n_permutations
        Total number of permutations to run across all workers.
    n_cores
        ``None`` (default) or ``1`` → serial, matches the R original's
        ``parallel=FALSE``. ``-1`` → all cores. ``>=2`` → that many
        worker processes via joblib's loky backend.
    seed
        Base seed. Each chunk gets its own spawned ``SeedSequence`` so the
        stacked null distribution is bit-for-bit identical regardless of
        how many workers are used.
    verbose
        Print a one-line summary of the chunking decision.
    """
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        return np.empty((0,))

    n_workers = _resolve_workers(n_cores)

    # Silent fall-back to serial if the workload wouldn't keep workers busy.
    if n_workers > 1 and n_permutations < n_workers * _MIN_PER_WORKER:
        if verbose:
            print(
                f"[parallel] {n_permutations} perms too few for {n_workers} "
                "workers — running serial"
            )
        n_workers = 1

    ss = np.random.SeedSequence(seed)

    if n_workers == 1:
        return chunk_fn(n_permutations, ss)

    # Same-sized chunks (last may be one shorter).
    chunk_sizes = [
        int(len(c)) for c in np.array_split(np.arange(n_permutations), n_workers)
    ]
    chunk_seeds = ss.spawn(len(chunk_sizes))
    if verbose:
        print(
            f"[parallel] {n_workers} workers × ~{chunk_sizes[0]} perms each "
            f"(total {n_permutations})"
        )

    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_workers)(
        delayed(chunk_fn)(cs, ss_i)
        for cs, ss_i in zip(chunk_sizes, chunk_seeds)
    )
    return np.concatenate(parts, axis=0)
