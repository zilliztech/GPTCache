"""A semantic cache front-end that drives GPTCache's real eviction policies.

The point of this module is that the benchmark measures *shipped code*. It
supplies only the half of the cache that GPTCache's vector store would supply
in production -- resident nearest-neighbour lookup -- and delegates every
eviction decision to a real
:class:`~gptcache.manager.eviction.memory_cache.MemoryCacheEviction`. So the
``LRU``/``LFU``/``FIFO``/``RR`` rows exercise ``cachetools`` exactly as
GPTCache does, and the ``ARC`` rows exercise
:class:`~gptcache.manager.eviction.arc.ARCCache` exactly as GPTCache does.

The request path mirrors ``SSDataManager``:

1. search the resident vectors; if the best cosine is ``>= tau`` it is a hit,
   and the policy is told via ``eviction_base.get(id)``;
2. otherwise it is a miss: the LLM would be called, and the new entry is
   admitted via ``eviction_base.put([id], embeddings=[vec])``;
3. the policy calls back through ``on_evict`` when entries leave, and this
   class drops their vectors -- standing in for the scalar and vector stores.
"""

import time

import numpy as np

from gptcache.manager.eviction.manager import EvictionBase


class SemanticCacheSim:
    """Semantic cache whose eviction is a real GPTCache policy.

    :param policy: ``LRU``, ``LFU``, ``FIFO``, ``RR`` or ``ARC``.
    :param capacity: resident capacity.
    :param tau: hit threshold, and (for ARC) the ghost-match threshold.
    :param dim: embedding width.
    :param kwargs: forwarded to the policy, e.g. ``ghost_matching``.
    """

    def __init__(self, policy, capacity, tau, dim, **kwargs):
        self.policy_name = policy
        self.cap = capacity
        self.tau = tau
        self.dim = dim
        self.next_id = 0

        # resident vectors, kept contiguous so lookup is one BLAS call
        self._m = np.zeros((capacity, dim), dtype=np.float32)
        self._row_of = {}
        self._id_at = []
        self._cid = {}

        if policy == "ARC":
            # ARC matches ghosts at the same threshold the cache serves at
            kwargs.setdefault("tau", tau)
        self.eviction = EvictionBase.get(
            name="memory", policy=policy, maxsize=capacity,
            clean_size=kwargs.pop("clean_size", 0),
            on_evict=self._on_evict, **kwargs
        )
        self.evicted = 0
        self.evict_seconds = 0.0
        self._evict_t0 = None

    # ------------------------------------------------------------------
    @property
    def raw_policy(self):
        """The underlying policy object (``ARCCache`` or a cachetools cache)."""
        return self.eviction._cache  # pylint: disable=protected-access

    def _on_evict(self, ids):
        for i in ids:
            self.evicted += 1
            row = self._row_of.pop(i, None)
            if row is None:
                continue
            last = len(self._id_at) - 1
            if row != last:
                moved = self._id_at[last]
                self._m[row] = self._m[last]
                self._id_at[row] = moved
                self._row_of[moved] = row
            self._id_at.pop()
            self._cid.pop(i, None)

    def _admit(self, vec, cid):
        key = self.next_id
        self.next_id += 1
        # the policy may evict synchronously inside put(); time that separately
        self._evict_t0 = time.perf_counter()
        if self.policy_name == "ARC":
            self.eviction.put([key], embeddings=[vec])
        else:
            self.eviction.put([key])
        self.evict_seconds += time.perf_counter() - self._evict_t0

        n = len(self._id_at)
        if n < self.cap:
            self._m[n] = vec
            self._row_of[key] = n
            self._id_at.append(key)
            self._cid[key] = cid
        else:
            # policy admitted an entry without releasing one: only possible if
            # it declined to evict, which none of these policies do
            raise RuntimeError(
                f"{self.policy_name} admitted an entry with the cache full "
                f"({n}/{self.cap}) and evicted nothing"
            )

    def request(self, vec, cid):
        """Serve one query.

        :returns: ``(hit, false_hit)``. ``false_hit`` is a hit served from an
            entry whose ground-truth cluster differs from the query's -- a
            wrong answer returned to the user. Only meaningful where cluster
            ids exist (``cid >= 0``).
        """
        n = len(self._id_at)
        if n:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = self._m[:n] @ vec
            j = int(sims.argmax())
            if sims[j] >= self.tau:
                key = self._id_at[j]
                self._evict_t0 = time.perf_counter()
                self.eviction.get(key)
                self.evict_seconds += time.perf_counter() - self._evict_t0
                stored = self._cid.get(key, -1)
                return True, (cid >= 0 and stored >= 0 and stored != cid)
        self._admit(vec, cid)
        return False, False

    # ------------------------------------------------------------------
    @property
    def p(self):
        """ARC's adaptive target size, or ``None`` for the other policies."""
        raw = self.raw_policy
        return getattr(raw, "p", None)

    @property
    def n_vectors(self):
        """Vectors the *policy* retains (ARC's ghosts), not the resident ones."""
        raw = self.raw_policy
        return getattr(raw, "n_vectors", 0)
