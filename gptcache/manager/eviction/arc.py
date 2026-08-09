"""Adaptive Replacement Cache with semantic ghost lists.

Self-contained: standard library and numpy only, plus :class:`EvictionBase`.
"""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from gptcache.manager.eviction.base import EvictionBase


class _VecList:
    """An ordered id set whose members carry a unit-norm embedding.

    Insertion order is LRU order: the leftmost element is the least recently
    used. Vectors are held in one contiguous matrix so a similarity scan is a
    single BLAS call rather than a Python loop, and removal is O(1) by swapping
    the last row into the vacated slot.

    This is an implementation detail of :class:`ARCCache`; the four ARC lists
    (T1, T2, B1, B2) are each one of these.
    """

    __slots__ = ("_dim", "_m", "_row_of", "_id_at", "_order")

    def __init__(self, dim: int, capacity_hint: int = 8):
        self._dim = dim
        self._m = np.zeros((max(capacity_hint, 1), dim), dtype=np.float32)
        self._row_of = {}          # id -> row index into _m
        self._id_at = []           # row index -> id
        self._order = {}           # id -> None; dicts are insertion-ordered

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, key) -> bool:
        return key in self._order

    def __iter__(self):
        return iter(self._order)

    @property
    def n_vectors(self) -> int:
        """Live vectors held by this list (== ``len(self)``)."""
        return len(self._id_at)

    def add(self, key, vec: np.ndarray) -> None:
        """Insert ``key`` at the MRU end, or refresh it if already present."""
        if key in self._order:
            self._m[self._row_of[key]] = vec
            self.touch(key)
            return
        n = len(self._id_at)
        if n == self._m.shape[0]:
            grown = np.zeros((2 * n, self._dim), dtype=np.float32)
            grown[:n] = self._m
            self._m = grown
        self._m[n] = vec
        self._row_of[key] = n
        self._id_at.append(key)
        self._order[key] = None

    def touch(self, key) -> None:
        """Move ``key`` to the MRU end."""
        del self._order[key]
        self._order[key] = None

    def remove(self, key) -> np.ndarray:
        """Drop ``key`` and return its embedding. O(1)."""
        row = self._row_of.pop(key)
        vec = self._m[row].copy()
        last = len(self._id_at) - 1
        if row != last:
            moved = self._id_at[last]
            self._m[row] = self._m[last]
            self._id_at[row] = moved
            self._row_of[moved] = row
        self._id_at.pop()
        del self._order[key]
        return vec

    def pop_lru(self) -> Tuple[Any, np.ndarray]:
        """Remove and return the least recently used ``(id, embedding)``."""
        key = next(iter(self._order))
        return key, self.remove(key)

    def nearest(self, query: np.ndarray, tau: float):
        """Id of the most similar member, if its cosine is >= ``tau``.

        Inputs are unit-norm, so cosine similarity is a dot product.
        Complexity: O(len(self) * dim), one BLAS ``matvec``, no allocation
        beyond the score vector.
        """
        n = len(self._id_at)
        if n == 0:
            return None
        # errstate: on macOS, numpy 2.x float32 matmul through Accelerate raises
        # spurious divide-by-zero / overflow / invalid flags on finite,
        # unit-norm input. Verified against a float64 einsum of the same data:
        # agreement to 8e-8. Without this a cache emits bogus warnings on every
        # miss.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._m[:n] @ query
        j = int(sims.argmax())
        return self._id_at[j] if sims[j] >= tau else None


class ARCCache(EvictionBase):
    """eviction: Adaptive Replacement Cache with **semantic** ghost lists.

    ARC (Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache*,
    USENIX FAST 2003) splits the cache into a recency list ``T1`` (items seen
    once) and a frequency list ``T2`` (items seen more than once), and keeps two
    *ghost* lists ``B1`` and ``B2`` of recently evicted items. Ghosts hold no
    cached response -- only enough identity to recognise a returning item. A hit
    in ``B1`` says "we evicted that too eagerly from the recency side", so the
    target size ``p`` of ``T1`` grows; a hit in ``B2`` says the opposite and
    ``p`` shrinks. That feedback loop is the whole of ARC's self-tuning, and it
    needs no workload-specific parameter. The Case I-IV structure below follows
    Figure 4 of that paper.

    **What is different here.** Classic ARC tests ghost membership by exact key.
    In a semantic cache that test can never succeed. GPTCache mints a fresh row
    id for every miss, and the queries that *should* recognise each other are
    paraphrases -- distinct strings, distinct ids. So ``B1`` and ``B2`` never
    fire, ``p`` never leaves 0, and ARC's adaptive machinery is silently dead
    rather than merely suboptimal. This class replaces every ghost membership
    test with a nearest-neighbour search over the ghosts' embeddings, accepting
    a match at cosine similarity ``>= tau``. Resident lookup is *not* done here:
    GPTCache's vector store already finds the resident match and then calls
    :meth:`get` with its id.

    Measured on real embeddings, the exact-key variant leaves ``p`` at exactly
    0 for an entire trace, while the semantic variant adapts and gains
    11-18 percentage points of hit rate on drifting traffic. Set
    ``ghost_matching="exact"`` to reproduce that ablation.

    **Costs, stated plainly.**

    - Memory: ghosts hold an embedding but no response, so the cache carries at
      most ``2 * maxsize`` vectors. At ``maxsize=1000`` and 384-d float32 that
      is about 3 MB.

      That is *not* a 2x memory cost, because a ghost is the embedding alone
      while a resident entry also carries the question and the cached response.
      The charge is ``1 + ghost/resident``. Measured over WildChat -- 1536 B
      embedding, 346 B question, 1531 B response -- a resident entry is 3414 B
      and ARC(c) costs **1.45x** what a ghost-free policy costs at the same
      capacity. It approaches 2x only as the cached responses approach empty.
      ``benchmarks/measure_payload.py`` recomputes this for a given corpus, and
      the benchmark reports hit rate against bytes on that basis; compare
      policies at equal memory, not at equal entry count.
    - Time: a miss scans both ghost lists, O(``2 * maxsize``) similarity
      computations, versus LRU's O(1) ``popitem``. A hit costs nothing extra.
      The benchmark harness measures this rather than hiding it.

    :param maxsize: resident capacity ``c`` -- the number of live cache entries.
    :type maxsize: int
    :param tau: cosine similarity at or above which a ghost is considered a
        match. Should be the same threshold the cache's similarity evaluation
        uses. Higher is stricter: fewer ghost hits and slower adaptation.
    :type tau: float
    :param ghost_matching: ``"semantic"`` (default) matches ghosts by embedding
        similarity; ``"exact"`` matches by id, reproducing classic ARC. The
        latter exists for the ablation and is not useful in production.
    :type ghost_matching: str
    :param on_evict: called with a list of ids as they leave the cache. Fires
        once per id, at the moment the entry is evicted -- never again when the
        id's ghost is later dropped.
    :type on_evict: Callable[[List[Any]], None]

    Example:
        .. code-block:: python

            import numpy as np
            from gptcache.manager.eviction.arc import ARCCache

            cache = ARCCache(maxsize=100, tau=0.8,
                             on_evict=lambda ids: print("evicted", ids))
            embedding = np.random.rand(384).astype("float32")
            embedding /= np.linalg.norm(embedding)
            cache.put([1], embeddings=[embedding])   # on a miss
            cache.get(1)                             # on a resident hit

    .. note::
        Embeddings are expected to be unit-norm, which is what
        ``SSDataManager.import_data`` already guarantees -- it normalises before
        storing. ``put`` may be called without embeddings (``SSDataManager``
        does this once at start-up to register rows already in the database).
        Such entries participate fully in eviction; they simply can never
        produce a ghost hit, degrading to classic ARC for those ids alone.
    """

    def __init__(
            self,
            maxsize: int = 1000,
            tau: float = 0.8,
            ghost_matching: str = "semantic",
            on_evict: Callable[[List[Any]], None] = None,
            dim: Optional[int] = None,
            **kwargs,
    ):
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if ghost_matching not in ("semantic", "exact"):
            raise ValueError(
                f"ghost_matching must be 'semantic' or 'exact', "
                f"got {ghost_matching!r}"
            )
        self._c = int(maxsize)
        self._tau = float(tau)
        self._semantic = ghost_matching == "semantic"
        self._ghost_matching = ghost_matching
        self._on_evict = on_evict
        self._p = 0.0
        self._dim = dim
        # True while _dim was guessed rather than observed: see _redim
        self._dim_provisional = False
        # created lazily: the embedding width is not known until the first put
        self._t1 = self._t2 = self._b1 = self._b2 = None
        if dim is not None:
            self._init_lists(dim)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _init_lists(self, dim: int, provisional: bool = False) -> None:
        hint = min(self._c + 1, 1024)
        self._dim = dim
        self._dim_provisional = provisional
        self._t1 = _VecList(dim, hint)
        self._t2 = _VecList(dim, hint)
        self._b1 = _VecList(dim, hint)
        self._b2 = _VecList(dim, hint)

    def _redim(self, dim: int) -> None:
        """Rebuild the lists at the true embedding width, preserving ids.

        Reached only from the provisional path: ids were admitted before any
        embedding had been seen (``SSDataManager`` registers the rows already
        in the database at start-up, without vectors), so the lists were
        created at a placeholder width. Every vector they hold is therefore a
        zero vector and carries no information -- rebuilding loses nothing but
        the placeholder width. Ids and their LRU order are preserved, so those
        entries keep evicting normally and simply never win a ghost match.
        """
        order = [list(lst)
                 for lst in (self._t1, self._t2, self._b1, self._b2)]
        self._init_lists(dim)
        zero = np.zeros(dim, dtype=np.float32)
        for ids, lst in zip(order,
                            (self._t1, self._t2, self._b1, self._b2)):
            for key in ids:
                lst.add(key, zero)

    def _coerce(self, embedding) -> np.ndarray:
        """Normalise an incoming embedding to a 1-D unit-norm float32 vector.

        A missing embedding becomes the zero vector, whose similarity to
        anything is 0 -- below any sensible ``tau`` -- so it can never win a
        ghost match. That is the intended degradation, not a silent failure.
        """
        if embedding is None:
            if self._dim is None:
                return None
            return np.zeros(self._dim, dtype=np.float32)
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if self._dim is None:
            self._init_lists(vec.shape[0])
        elif vec.shape[0] != self._dim:
            if self._dim_provisional:
                self._redim(vec.shape[0])
            else:
                raise ValueError(
                    f"embedding has dimension {vec.shape[0]}, "
                    f"expected {self._dim}"
                )
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _evict(self, key) -> None:
        """Announce that ``key``'s entry has left the cache. Fires once."""
        if self._on_evict is not None:
            self._on_evict([key])

    def _match(self, ghosts: _VecList, query, key):
        """Ghost membership test -- the semantic modification lives here.

        ``"semantic"``: the nearest ghost by cosine similarity, if it clears
        ``tau``. ``"exact"``: plain id membership, i.e. classic ARC.

        :returns: the matching ghost id, or ``None``.
        """
        if not self._semantic:
            return key if key in ghosts else None
        if query is None:
            return None
        return ghosts.nearest(query, self._tau)

    def _replace(self, in_b2: bool) -> None:
        """ARC's REPLACE subroutine: demote one resident entry to a ghost.

        The entry leaves the cache (so ``on_evict`` fires) but its embedding is
        retained on the corresponding ghost list.
        """
        if len(self._t1) > 0 and (
            len(self._t1) > self._p or (in_b2 and len(self._t1) == self._p)
        ):
            key, vec = self._t1.pop_lru()
            self._b1.add(key, vec)
        elif len(self._t2) > 0:
            key, vec = self._t2.pop_lru()
            self._b2.add(key, vec)
        else:
            return
        self._evict(key)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get(self, obj: Any):
        """Record a hit on a resident entry (ARC Case I).

        An entry hit while in ``T1`` has now been seen more than once, so it is
        promoted to the frequency list ``T2``; an entry hit while already in
        ``T2`` is refreshed to the MRU end. Ghost lists are untouched -- a ghost
        is by definition not resident, and GPTCache only calls this after its
        vector store has found a live match.

        :param obj: the entry id.
        :returns: ``True`` if the id was resident, otherwise ``None`` -- the
            same convention as the ``cachetools``-backed policies.
        :complexity: O(1).
        """
        if self._t1 is None:
            return None
        if obj in self._t1:
            vec = self._t1.remove(obj)
            self._t2.add(obj, vec)
            return True
        if obj in self._t2:
            self._t2.touch(obj)
            return True
        return None

    def put(self, objs: List[Any], embeddings: Optional[List[Any]] = None):
        """Admit entries after a cache miss (ARC Cases II, III and IV).

        For each id, the ghost lists are consulted first. A hit in ``B1`` means
        recency was the right signal and ``p`` grows; a hit in ``B2`` means
        frequency was, and ``p`` shrinks. Either way the new entry enters the
        frequency list ``T2``, because something like it has been seen before.
        With no ghost hit this is a genuinely new item and it enters the recency
        list ``T1``.

        :param objs: entry ids to admit.
        :param embeddings: matching unit-norm embeddings, one per id. Optional:
            when absent the ids are admitted without ghost-matching ability
            (see the class note).
        :complexity: O(``maxsize`` * ``dim``) per id -- one similarity scan of
            each ghost list. O(1) once the lists are exhausted of vectors, i.e.
            under ``ghost_matching="exact"``.
        """
        if embeddings is not None and len(embeddings) != len(objs):
            raise ValueError(
                f"embeddings has length {len(embeddings)}, expected "
                f"{len(objs)} to match objs"
            )
        for i, key in enumerate(objs):
            raw = embeddings[i] if embeddings is not None else None
            self._put_one(key, self._coerce(raw))

    def _put_one(self, key, query) -> None:
        if self._t1 is None:
            # no embedding has ever been seen; fall back to a scalar cache of
            # 1-d zero vectors so ids can still be admitted and evicted. The
            # width is provisional -- the first real embedding re-dimensions
            # the lists rather than being rejected as a mismatch.
            self._init_lists(1, provisional=True)
            query = np.zeros(1, dtype=np.float32)
        if query is None:
            query = np.zeros(self._dim, dtype=np.float32)

        # degenerate: re-putting a resident id is an access, not an admission
        if key in self._t1 or key in self._t2:
            self.get(key)
            return
        c, b1, b2 = self._c, self._b1, self._b2
        gb1 = self._match(b1, query, key)
        gb2 = self._match(b2, query, key) if gb1 is None else None
        # A returning id that is its own ghost must not end up on two lists at
        # once. Cases II and III already retire the ghost they matched, so only
        # an *unmatched* self-ghost needs retiring here -- and it must happen
        # after the match, not before: retiring first would delete the very
        # ghost that exact matching looks for, making Cases II and III
        # unreachable and pinning p at 0 for every workload, not just the
        # fresh-id ones where that outcome is the genuine finding.
        if key in b1 and key != gb1:
            b1.remove(key)
        elif key in b2 and key != gb2:
            b2.remove(key)

        if gb1 is not None:
            # Case II -- recency was right, grow the target size of T1
            delta = max(1.0, len(b2) / max(len(b1), 1))
            self._p = min(float(c), self._p + delta)
            self._replace(in_b2=False)
            b1.remove(gb1)
            self._t2.add(key, query)
        elif gb2 is not None:
            # Case III -- frequency was right, shrink the target size of T1
            delta = max(1.0, len(b1) / max(len(b2), 1))
            self._p = max(0.0, self._p - delta)
            self._replace(in_b2=True)
            b2.remove(gb2)
            self._t2.add(key, query)
        else:
            # Case IV -- a genuinely new item
            l1 = len(self._t1) + len(b1)
            if l1 == c:
                if len(self._t1) < c:
                    b1.pop_lru()
                    self._replace(in_b2=False)
                else:
                    evicted, _ = self._t1.pop_lru()
                    self._evict(evicted)
            elif l1 < c and l1 + len(self._t2) + len(b2) >= c:
                if l1 + len(self._t2) + len(b2) >= 2 * c:
                    b2.pop_lru()
                self._replace(in_b2=False)
            self._t1.add(key, query)

    @property
    def policy(self) -> str:
        # required by EvictionBase; MemoryCacheEviction reports its own name,
        # so this is what identifies an ARCCache held directly
        return "ARC"

    # ------------------------------------------------------------------
    # introspection -- used by the tests, the ablation and the benchmarks
    # ------------------------------------------------------------------
    @property
    def p(self) -> float:
        """Adaptive target size for ``T1``.

        Stays at 0 under ``ghost_matching="exact"`` on any workload that mints
        a fresh id per miss -- which is every GPTCache workload, and the point
        of the ablation. Exact matching still adapts normally on stable ids.
        """
        return self._p

    @property
    def maxsize(self) -> int:
        """Resident capacity ``c``. Ghosts are held on top of this."""
        return self._c

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def ghost_matching(self) -> str:
        return self._ghost_matching

    @property
    def sizes(self) -> dict:
        """Current ``|T1| |T2| |B1| |B2|``."""
        if self._t1 is None:
            return {"T1": 0, "T2": 0, "B1": 0, "B2": 0}
        return {"T1": len(self._t1), "T2": len(self._t2),
                "B1": len(self._b1), "B2": len(self._b2)}

    @property
    def n_vectors(self) -> int:
        """Live embeddings retained. Bounded by ``2 * maxsize``."""
        if self._t1 is None:
            return 0
        return sum(lst.n_vectors
                   for lst in (self._t1, self._t2, self._b1, self._b2))

    def resident_ids(self) -> List[Any]:
        """Ids currently holding a cached response (``T1`` then ``T2``)."""
        if self._t1 is None:
            return []
        return list(self._t1) + list(self._t2)

    def list_ids(self) -> dict:
        """Ids held by each of the four lists, for invariant checking.

        ``tests/unit_tests/eviction/test_arc_cache.py`` asserts Megiddo & Modha
        section III.B over this after every operation in its fuzz tests.
        """
        if self._t1 is None:
            return {"T1": [], "T2": [], "B1": [], "B2": []}
        return {"T1": list(self._t1), "T2": list(self._t2),
                "B1": list(self._b1), "B2": list(self._b2)}
