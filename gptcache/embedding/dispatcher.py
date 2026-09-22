import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, List, Optional

from gptcache.embedding.base import BaseEmbedding


# ---------------------------------------------------------------------------
# Worker-process globals. Each worker process builds its own model exactly
# once (in _init_worker, run at process start) and reuses it for every task
# it's given -- the model is NOT rebuilt per call.
# ---------------------------------------------------------------------------
_worker_model = None


def _init_worker(embedding_factory):
    """Runs once, in each worker process, when the process pool starts."""
    global _worker_model
    _worker_model = embedding_factory()


def _embed_task(data):
    """Runs in a worker process. Uses the model built by _init_worker."""
    return _worker_model.to_embeddings(data)


class EmbeddingDispatcher(BaseEmbedding):
    """Fan per-request embedding calls across multiprocessing worker
    processes.

    Targets embedding *throughput* under concurrent load, not single-call
    latency -- a lone caller sees only IPC overhead and is slower. Measured on
    an SBERT/MiniLM CPU encoder over UltraChat (3 runs, see
    examples/benchmark/benchmark_embedding_dispatcher.py): the dispatcher loses
    below ~50 concurrent callers (process dispatch + per-task IPC exceed the
    per-encode work) and wins up to ~2x at 50-100, where one shared model under
    many threads contends but 8 independent worker processes do not. The cost
    is memory: one model copy per worker, a ~5 GB RSS floor paid at pool
    creation regardless of load, vs ~1 GB sequential. A throughput-vs-memory
    Pareto point -- worth it only for a server fielding many overlapping
    requests. Measured on one machine; the crossover point is host-dependent.

    IMPORTANT (Windows / pickling): embedding_factory must be a
    picklable, zero-argument callable -- a module-level function or a
    callable class instance, NOT a lambda or closure. On Windows,
    multiprocessing uses the "spawn" start method, which pickles the
    factory to hand it to each new worker process; lambdas cannot be
    pickled and will raise at pool-creation time, not silently misbehave.

    Example:
        def make_sbert():
            from gptcache.embedding import SBERT
            return SBERT("all-MiniLM-L6-v2")

        dispatcher = EmbeddingDispatcher(make_sbert, num_workers=4)
        embed = dispatcher.to_embeddings("Hello, world.")
        dispatcher.shutdown()
    """

    def __init__(
        self,
        embedding_factory,
        num_workers=None,
        batch_mode=False,
    ):
        if num_workers is None:
            cpu = os.cpu_count() or 2
            num_workers = max(1, min(cpu - 1, 8))
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")

        self._num_workers = num_workers
        self._batch_mode = batch_mode
        self._dimension = None
        self._pool = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(embedding_factory,),
        )
        self._closed = False

    def to_embeddings(self, data, **_):
        """Blocking call: submit to the worker pool and wait for the
        result. Safe to call from multiple threads in the caller's process
        simultaneously -- that's how concurrent load actually reaches the
        worker processes in parallel (each thread blocks on its own
        future while the pool runs them across processes)."""
        return self.to_embeddings_async(data).result()

    def to_embeddings_async(self, data):
        """Non-blocking variant: returns a concurrent.futures.Future
        immediately. Fan out many calls, then collect results, to exercise
        real cross-process parallelism from a single caller thread."""
        if self._closed:
            raise RuntimeError("EmbeddingDispatcher is shut down")
        return self._pool.submit(_embed_task, data)

    def to_embeddings_batch(self, data_list, **_):
        """Submit a batch of independent texts, one task per item, and
        wait for all results."""
        futures = [self.to_embeddings_async(d) for d in data_list]
        return [f.result() for f in futures]

    @property
    def dimension(self):
        if self._dimension is None:
            self._dimension = len(self.to_embeddings("foo"))
        return self._dimension

    @property
    def num_workers(self):
        return self._num_workers

    def shutdown(self, wait=True):
        """Terminate all worker processes. Safe to call more than once."""
        if not self._closed:
            self._pool.shutdown(wait=wait)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except Exception:
            pass
