import time

import numpy as np
import pytest

from gptcache.embedding.base import BaseEmbedding
from gptcache.embedding.dispatcher import EmbeddingDispatcher


class _SlowFakeEmbedding(BaseEmbedding):
    """Deterministic fake encoder with an artificial delay, so a test can
    tell serial-in-one-process apart from parallel-across-processes by
    wall-clock time."""

    def __init__(self, dim=8, latency_s=0.05):
        self._dim = dim
        self._latency_s = latency_s

    def to_embeddings(self, data, **_):
        time.sleep(self._latency_s)
        seed = abs(hash(data)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self._dim).astype("float32")

    @property
    def dimension(self):
        return self._dim


def _make_slow_fake_embedding():
    return _SlowFakeEmbedding(dim=8, latency_s=0.05)


def test_single_call_returns_correct_dimension():
    with EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=2) as d:
        emb = d.to_embeddings("hello")
        assert emb.shape == (8,)
        assert d.dimension == 8


def test_distinct_calls_return_deterministic_values():
    with EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=2) as d:
        a1 = d.to_embeddings("alpha")
        a2 = d.to_embeddings("alpha")
        b = d.to_embeddings("beta")
        assert np.array_equal(a1, a2)
        assert not np.array_equal(a1, b)


def test_concurrent_calls_are_faster_than_serial_equivalent():
    """The whole point of the dispatcher: N calls across >1 workers should
    take meaningfully less wall-clock time than N * per-call latency.

    We warm up the pool first (one call per worker) so the one-time cost of
    starting worker processes -- expensive on Windows, which uses "spawn"
    and re-imports the interpreter per process -- isn't counted against the
    steady-state parallel speedup. This matches real deployment, where
    workers start once at boot, not per request."""
    n_workers = 4
    n_calls = 8
    per_call_latency = 0.05

    with EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=n_workers) as d:
        warmup = [d.to_embeddings_async(f"warmup-{i}") for i in range(n_workers)]
        for f in warmup:
            f.result()

        futures = [d.to_embeddings_async(f"query-{i}") for i in range(n_calls)]
        start = time.perf_counter()
        for f in futures:
            f.result()
        elapsed = time.perf_counter() - start

    serial_estimate = n_calls * per_call_latency
    assert elapsed < serial_estimate


def test_batch_helper_returns_results_in_order():
    with EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=3) as d:
        results = d.to_embeddings_batch(["a", "b", "c"])
        assert len(results) == 3
        assert np.array_equal(results[0], d.to_embeddings("a"))


def test_shutdown_is_idempotent():
    d = EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=2)
    d.to_embeddings("warm up")
    d.shutdown()
    d.shutdown()


def test_call_after_shutdown_raises():
    d = EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=2)
    d.shutdown()
    with pytest.raises(RuntimeError):
        d.to_embeddings_async("too late")


def test_invalid_num_workers_raises():
    with pytest.raises(ValueError):
        EmbeddingDispatcher(_make_slow_fake_embedding, num_workers=0)


def test_lambda_factory_is_not_picklable():
    """Documents and enforces the Windows-relevant constraint: a lambda
    factory cannot be pickled to send to worker processes under the
    "spawn" start method, which Windows always uses (Linux/macOS default
    to "fork", which doesn't need pickling)."""
    import pickle

    with pytest.raises((TypeError, pickle.PicklingError, AttributeError)):
        pickle.dumps(lambda: _SlowFakeEmbedding())

    pickle.dumps(_make_slow_fake_embedding)
