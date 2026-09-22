"""Phase 3 smoke test: CA_W_TINYLFU eviction wired into SSDataManager.

Verifies end-to-end behaviour without any LLM or network calls:
  1. CA_W_TINYLFU routes through get_data_manager without crashing.
  2. Inserting more items than maxsize triggers the on_evict callback.
  3. The eviction base stays within maxsize after the overflow.
  4. Evicted IDs are no longer tracked in the eviction base.

Run from the repo root:
    python examples/smoke/smoke_ca_w_tinylfu.py
"""

import os
import tempfile

import numpy as np

from gptcache.manager import get_data_manager, CacheBase, VectorBase

DIMENSION = 8
MAX_SIZE = 50
N_INSERTS = 200


def _rand_embedding() -> np.ndarray:
    v = np.random.randn(DIMENSION).astype("float32")
    return v / np.linalg.norm(v)


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "smoke.db")
        index_path = os.path.join(tmpdir, "smoke.index")

        # --- 1. Build SSDataManager with CA_W_TINYLFU ---
        data_manager = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{db_path}"),
            VectorBase("faiss", dimension=DIMENSION, index_path=index_path),
            max_size=MAX_SIZE,
            eviction="CA_W_TINYLFU",
        )

        # Intercept on_evict to record evicted IDs while still letting
        # SSDataManager._clear handle the SQLite/Faiss cleanup.
        evicted_ids: list = []
        _cache = data_manager.eviction_base._cache   # CostAwareWTinyLFU instance
        _original_on_evict = _cache._on_evict

        def _tracking_on_evict(keys: list) -> None:
            evicted_ids.extend(keys)
            _original_on_evict(keys)

        _cache._on_evict = _tracking_on_evict

        # --- 2. Insert N_INSERTS unique items ---
        print(f"Inserting {N_INSERTS} items into a cache of maxsize={MAX_SIZE} ...")
        for i in range(N_INSERTS):
            data_manager.save(
                question=f"smoke_question_{i}",
                answer=f"smoke_answer_{i}",
                embedding_data=_rand_embedding(),
            )

        # --- 3. Assertions ---
        cache_len = len(_cache)
        n_evicted = len(evicted_ids)
        print(f"  cache size : {cache_len} / {MAX_SIZE}")
        print(f"  evictions  : {n_evicted}")

        assert n_evicted >= 1, (
            f"on_evict never fired despite inserting {N_INSERTS} > {MAX_SIZE} items"
        )
        assert cache_len <= MAX_SIZE, (
            f"cache size {cache_len} exceeds maxsize {MAX_SIZE}"
        )
        for eid in evicted_ids:
            assert data_manager.eviction_base.get(eid) is None, (
                f"evicted ID {eid} still returned by eviction_base.get()"
            )

        data_manager.close()

    print("\nPhase 3 smoke test PASSED.")


if __name__ == "__main__":
    main()
