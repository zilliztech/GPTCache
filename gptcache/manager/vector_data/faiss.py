import os
from typing import List, Optional, Union

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_faiss
from gptcache.utils.log import gptcache_log

import_faiss()

import faiss  # pylint: disable=C0413


class Faiss(VectorBase):
    """vector store: Faiss

    Supports multiple index types for different performance trade-offs:

    - ``"flat"`` (default): Exact brute-force search with ``IDMap,Flat``.
      Best recall, O(n) search. No training needed.
    - ``"hnsw_sq8"``: HNSW graph index with 8-bit scalar quantization,
      wrapped in ``IndexIDMap`` for custom ID support.
      ~4x memory reduction vs Flat, O(log n) search, high recall.
      **Does not support per-vector deletion** — uses tombstone marking
      and periodic rebuild instead.
    - ``"hnsw_pq"``: HNSW graph index with Product Quantization
      (``PQ{m_pq}x8`` → ``m_pq`` bytes per code). Far smaller than SQ8
      (e.g. 32 B/vec vs ``dim`` B/vec) at the cost of recall. Needs a real
      training step on ``pq_train_size`` representative vectors, so the
      first adds are buffered until that many vectors arrive (see ``mul_add``).
    - ``"hnsw_pq_refine"``: ``hnsw_pq`` wrapped in ``IndexRefineFlat`` so the
      top ``k_factor·k`` PQ candidates are re-ranked against full-precision
      vectors. Recovers most of PQ's recall loss, but **keeps the full
      vectors in RAM**, so it trades PQ's compression back for quality — a
      Pareto point, not a free lunch.

    All HNSW variants share the tombstone deletion path (no structural
    ``remove_ids``).

    :param index_path: the path to Faiss index, defaults to 'faiss.index'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param index_type: index type, one of ``"flat"``, ``"hnsw_sq8"``,
                       ``"hnsw_pq"`` or ``"hnsw_pq_refine"``, defaults to ``"flat"``.
    :type index_type: str
    :param hnsw_m: number of links per node in HNSW graph (higher = better recall, more memory),
                   defaults to 32.
    :type hnsw_m: int
    :param hnsw_ef_construction: size of the dynamic candidate list during construction
                                 (higher = better recall, slower build), defaults to 200.
    :type hnsw_ef_construction: int
    :param hnsw_ef_search: size of the dynamic candidate list during search
                           (higher = better recall, slower search), defaults to 128.
    :type hnsw_ef_search: int
    :param m_pq: number of PQ sub-quantizers (= bytes per code) for the
                 ``hnsw_pq``/``hnsw_pq_refine`` types, defaults to 32.
    :type m_pq: int
    :param k_factor: ``IndexRefineFlat`` over-fetch factor for ``hnsw_pq_refine``
                     (re-rank ``k_factor·k`` candidates), defaults to 4.
    :type k_factor: int
    :param pq_train_size: number of vectors to buffer before training PQ. ``None``
                          → ``max(50 * m_pq, 2048)``. Ignored for non-PQ types.
    :type pq_train_size: int
    """

    def __init__(
        self,
        index_file_path,
        dimension,
        top_k,
        index_type="flat",
        hnsw_m=32,
        hnsw_ef_construction=200,
        # Default ef=64: a 10K sweep showed ef=32 holds true-positive rate,
        # so 64 leaves ~4x headroom. Override per call via search(ef_search=...).
        hnsw_ef_search=64,
        m_pq=32,
        k_factor=4,
        pq_train_size=None,
    ):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._top_k = top_k
        self._index_type = index_type.lower()
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._m_pq = m_pq
        self._k_factor = k_factor
        self._pq_train_size = pq_train_size or max(50 * m_pq, 2048)

        # Convenience flags: all hnsw_* variants share the tombstone path; the
        # PQ variants additionally need a buffered training warmup.
        self._is_hnsw = self._index_type.startswith("hnsw")
        self._is_pq = self._index_type in ("hnsw_pq", "hnsw_pq_refine")
        # PQ training buffer: adds are held here until pq_train_size vectors
        # arrive, then trained-on and flushed in one shot (see mul_add).
        self._train_buffer = []  # list of (np_data, ids)
        self._train_buffered = 0

        # For HNSW: tombstone set of deleted IDs (since HNSW can't do remove_ids)
        self._tombstones = set()

        if os.path.isfile(index_file_path):
            self._index = faiss.read_index(index_file_path)
            # Restore tombstones if saved alongside the index
            tombstone_path = index_file_path + ".tombstones.npy"
            if os.path.isfile(tombstone_path):
                self._tombstones = set(np.load(tombstone_path).tolist())
            gptcache_log.info(
                "Loaded existing Faiss index from %s (ntotal=%d, tombstones=%d)",
                index_file_path,
                self._index.ntotal,
                len(self._tombstones),
            )
        else:
            self._index = self._create_index(
                dimension, self._index_type, hnsw_m, hnsw_ef_construction,
                hnsw_ef_search, self._m_pq, self._k_factor,
            )

    @staticmethod
    def _create_index(dimension, index_type, hnsw_m=32, hnsw_ef_construction=200,
                      hnsw_ef_search=128, m_pq=32, k_factor=4):
        """Create a new FAISS index of the specified type.

        :param dimension: vector dimensionality.
        :param index_type: ``"flat"``, ``"hnsw_sq8"``, ``"hnsw_pq"`` or ``"hnsw_pq_refine"``.
        :param hnsw_m: HNSW M parameter.
        :param hnsw_ef_construction: HNSW efConstruction parameter.
        :param hnsw_ef_search: HNSW efSearch parameter.
        :param m_pq: PQ sub-quantizer count (bytes/code) for the PQ variants.
        :param k_factor: ``IndexRefineFlat`` over-fetch factor for ``hnsw_pq_refine``.
        :return: a configured ``faiss.Index`` wrapped in ``IndexIDMap``.
        """
        if index_type == "hnsw_sq8":
            # HNSW graph with 8-bit Scalar Quantization
            # - HNSW{M}: graph connectivity (higher M = better recall, more memory)
            # - SQ8: each float32 compressed to uint8 (4x memory savings)
            factory_string = f"HNSW{hnsw_m},SQ8"
            base_index = faiss.index_factory(dimension, factory_string, faiss.METRIC_L2)

            # Set HNSW-specific parameters for recall/speed trade-off
            hnsw_index = faiss.downcast_index(base_index)
            hnsw_index.hnsw.efSearch = hnsw_ef_search
            hnsw_index.hnsw.efConstruction = hnsw_ef_construction

            # Wrap in IndexIDMap so we can use add_with_ids (custom IDs)
            # HNSW natively uses sequential IDs; IDMap translates custom → internal
            index = faiss.IndexIDMap(base_index)

            # SQ8 requires a lightweight training step (learns min/max per dimension).
            # Unlike IVF+PQ, this is virtually instant and can be done on the first
            # batch of vectors — no cold start problem.
            gptcache_log.info(
                "Created HNSW+SQ8 index (dim=%d, M=%d, efConstruction=%d, efSearch=%d)",
                dimension, hnsw_m, hnsw_ef_construction, hnsw_ef_search,
            )
            return index
        elif index_type in ("hnsw_pq", "hnsw_pq_refine"):
            # HNSW graph over Product-Quantized codes.
            # - PQ{m_pq}x8: each vector is m_pq bytes (m_pq sub-quantizers, 8 bits
            #   each), i.e. m_pq B/vec vs dim B/vec for SQ8 — a big compression win
            #   at the cost of recall. PQ needs a real training step (see mul_add).
            factory_string = f"HNSW{hnsw_m},PQ{m_pq}x8"
            base_index = faiss.index_factory(dimension, factory_string, faiss.METRIC_L2)

            hnsw_index = faiss.downcast_index(base_index)
            hnsw_index.hnsw.efSearch = hnsw_ef_search
            hnsw_index.hnsw.efConstruction = hnsw_ef_construction

            if index_type == "hnsw_pq_refine":
                # IndexRefineFlat re-ranks the top k_factor*k PQ candidates against
                # full-precision vectors. This recovers recall but keeps the float32
                # vectors in RAM, so it gives back PQ's compression — report it as a
                # Pareto point, not a free win.
                refine = faiss.IndexRefineFlat(base_index)
                refine.k_factor = k_factor
                index = faiss.IndexIDMap(refine)
                gptcache_log.info(
                    "Created HNSW+PQ+Refine index (dim=%d, M=%d, PQ=%dx8, k_factor=%d)",
                    dimension, hnsw_m, m_pq, k_factor,
                )
            else:
                index = faiss.IndexIDMap(base_index)
                gptcache_log.info(
                    "Created HNSW+PQ index (dim=%d, M=%d, PQ=%dx8)",
                    dimension, hnsw_m, m_pq,
                )
            return index
        else:
            # Default: exact brute-force with ID mapping
            return faiss.index_factory(dimension, "IDMap,Flat", faiss.METRIC_L2)

    @property
    def index_type(self):
        return self._index_type

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)

        if self._is_pq and not self._index.is_trained:
            # PQ needs a real training step on many representative vectors
            # (256 centroids per sub-quantizer). Buffer the first adds until we
            # have pq_train_size of them, then train once and flush the buffer.
            self._train_buffer.append((np_data, ids))
            self._train_buffered += len(np_data)
            if self._train_buffered >= self._pq_train_size:
                self._train_and_flush_buffer()
            return

        if self._index_type == "hnsw_sq8" and not self._index.is_trained:
            # SQ8 training: learns per-dimension min/max for quantization.
            # This is virtually instant (unlike IVF+PQ which needs ~10K vectors).
            self._index.train(np_data)
            gptcache_log.info("Trained HNSW+SQ8 index on %d vectors", len(np_data))

        self._index.add_with_ids(np_data, ids)

    def _train_and_flush_buffer(self):
        """Train the PQ index on the buffered vectors, then add them all.

        Called once the buffer reaches ``pq_train_size`` (normal path) or lazily
        from ``search``/``flush`` if fewer vectors than that ever arrive, so a
        small dataset still becomes searchable. FAISS raises if the buffer holds
        too few points to train PQ — that surfaces a genuine misconfiguration
        rather than silently producing an untrained index.
        """
        if not self._train_buffer:
            return
        all_data = np.concatenate([d for d, _ in self._train_buffer]).astype("float32")
        all_ids = np.concatenate([i for _, i in self._train_buffer])
        self._index.train(all_data)
        self._index.add_with_ids(all_data, all_ids)
        gptcache_log.info(
            "Trained %s index on %d vectors and flushed buffer",
            self._index_type, len(all_data),
        )
        self._train_buffer = []
        self._train_buffered = 0

    def search(self, data: np.ndarray, top_k: int = -1, ef_search: int = None):
        """Search the index for the top-k nearest neighbours.

        :param data: query vector.
        :param top_k: how many neighbours to return; ``-1`` uses the configured default.
        :param ef_search: HNSW per-call ``efSearch`` override. Higher = more
            candidates explored = better recall at higher latency. Only meaningful
            for ``hnsw_sq8``; ignored for ``flat``. ``None`` keeps the index-level
            default set at construction time.
        """
        # If PQ data is still buffered (dataset smaller than pq_train_size),
        # train + flush now so it becomes searchable.
        if self._is_pq and not self._index.is_trained and self._train_buffer:
            self._train_and_flush_buffer()

        if self._index.ntotal == 0:
            return None
        if top_k == -1:
            top_k = self._top_k

        np_data = np.array(data).astype("float32").reshape(1, -1)

        # Per-call efSearch override - the cheapest knob in HNSW. Restore the
        # default after the call so concurrent searches with different settings
        # don't poison each other.
        hnsw = None
        prior_ef = None
        if self._is_hnsw and ef_search is not None:
            try:
                inner = faiss.downcast_index(self._index.index)  # unwrap IndexIDMap
                if self._index_type == "hnsw_pq_refine":
                    # One more layer: IndexIDMap -> IndexRefineFlat -> HNSW base.
                    inner = faiss.downcast_index(inner.base_index)
                hnsw = inner.hnsw
                prior_ef = hnsw.efSearch
                hnsw.efSearch = int(ef_search)
            except Exception:  # noqa: BLE001
                hnsw = None  # silently fall back to the index-level default

        try:
            if self._is_hnsw and self._tombstones:
                # Over-fetch to compensate for tombstoned results we'll filter out
                fetch_k = min(top_k + len(self._tombstones), self._index.ntotal)
                dist, ids = self._index.search(np_data, fetch_k)
                # Filter out tombstoned IDs
                results = []
                for d, i in zip(dist[0], ids[0]):
                    i = int(i)
                    if i == -1 or i in self._tombstones:
                        continue
                    results.append((d, i))
                    if len(results) >= top_k:
                        break
                return results if results else None
            else:
                dist, ids = self._index.search(np_data, top_k)
                ids = [int(i) for i in ids[0]]
                return list(zip(dist[0], ids))
        finally:
            if hnsw is not None and prior_ef is not None:
                hnsw.efSearch = prior_ef

    def rebuild(self, ids=None):
        """Rebuild the index, removing physically deleted vectors where possible.

        For flat: clears the (unused) tombstone set — physical removal was
        already done by ``remove_ids`` in ``delete()``.
        For hnsw (SQ8/PQ): the HNSW graph does not expose a decode/compaction
        path for the stored codes, so structural compaction is not possible
        here. Tombstones are intentionally **kept** so that evicted IDs continue
        to be filtered out of search results in all subsequent calls to
        ``search()``.

        The tombstone set is bounded by the total number of evictions over the
        cache lifetime (each eviction adds at most one entry). At 8 bytes per
        int64, even 1 million cumulative evictions costs only ~8 MB.
        """
        if self._is_hnsw:
            # HNSW cannot physically remove vectors. Tombstones remain active
            # and must NOT be cleared — clearing them would allow deleted
            # vectors to reappear in search results.
            return True
        self._tombstones.clear()
        return True

    def delete(self, ids):
        """Delete vectors by their IDs.

        For Flat index: uses FAISS native ``remove_ids``.
        For HNSW (SQ8/PQ): marks IDs as tombstones (logical deletion) since
        HNSW does not support structural deletion. Tombstoned IDs are
        filtered out during search and removed on the next ``rebuild()``.
        """
        if self._is_hnsw:
            # HNSW does not support remove_ids — use tombstone marking
            self._tombstones.update(int(i) for i in ids)
            gptcache_log.debug(
                "Tombstoned %d IDs in HNSW index (total tombstones: %d)",
                len(ids),
                len(self._tombstones),
            )
        else:
            ids_to_remove = np.array(ids, dtype=np.int64)
            self._index.remove_ids(faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove)))

    def flush(self):
        # Make sure any buffered PQ vectors are trained + added before persisting,
        # otherwise the on-disk index would be empty for sub-pq_train_size datasets.
        if self._is_pq and not self._index.is_trained and self._train_buffer:
            self._train_and_flush_buffer()
        faiss.write_index(self._index, self._index_file_path)
        tombstone_path = self._index_file_path + ".tombstones.npy"
        if self._tombstones:
            np.save(tombstone_path, np.array(list(self._tombstones)))
        elif os.path.isfile(tombstone_path):
            # Remove stale tombstone file left over from a previous flush so
            # that a subsequent load does not restore already-evicted IDs.
            os.remove(tombstone_path)

    def close(self):
        self.flush()

    def count(self):
        return self._index.ntotal
