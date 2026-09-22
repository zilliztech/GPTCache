import os
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData
from gptcache.manager.vector_data.docarray_index import DocArrayIndex
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.manager.vector_data.hnswlib_store import Hnswlib

DIM = 512
MAX_ELEMENTS = 10000
SIZE = 1000
TOP_K = 10


class TestLocalIndex(unittest.TestCase):
    def test_faiss(self):
        cls = partial(Faiss, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='faiss', top_k=3, dimension=DIM, index_path=index_path
            )

    def test_faiss_hnsw_sq8(self):
        """Test HNSW+SQ8 index: add, search, tombstone deletion, rebuild, persistence."""
        cls = partial(Faiss, dimension=DIM, index_type="hnsw_sq8")

        # --- Basic add and search ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            self.assertEqual(index.index_type, "hnsw_sq8")
            self.assertEqual(len(index.search(data[0])), TOP_K)
            # Nearest neighbor of data[0] should be itself (id=0)
            self.assertEqual(index.search(data[0])[0][1], 0)

        # --- Tombstone deletion filters results ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            # Delete id=0, search for data[0] should NOT return id=0
            index.delete([0])
            results = index.search(data[0])
            result_ids = [r[1] for r in results]
            self.assertNotIn(0, result_ids)
            # ntotal still includes tombstoned vectors
            self.assertEqual(index.count(), SIZE)

        # --- Rebuild preserves tombstones (HNSW cannot physically evict vectors) ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1, 2])
            self.assertEqual(len(index._tombstones), 3)
            index.rebuild(list(range(3, SIZE)))
            # Tombstones must NOT be cleared — HNSW keeps deleted vectors in the
            # graph; clearing the tombstone set would make them reappear in search.
            self.assertEqual(len(index._tombstones), 3)

        # --- Persistence: tombstones survive save/load ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1])
            index.close()  # flush index + tombstones to disk

            # Reload and verify tombstones were restored
            new_index = cls(index_file_path=index_path, top_k=TOP_K)
            self.assertEqual(len(new_index._tombstones), 2)
            results = new_index.search(data[0])
            result_ids = [r[1] for r in results]
            self.assertNotIn(0, result_ids)
            self.assertNotIn(1, result_ids)

        # --- Rebuild keeps deleted IDs filtered in search ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1, 2])
            index.rebuild(list(range(3, SIZE)))
            # Tombstones stay — deleted IDs must not appear in search results
            self.assertEqual(len(index._tombstones), 3)
            results = index.search(data[0])
            result_ids = [r[1] for r in results]
            self.assertNotIn(0, result_ids, "id=0 must remain filtered after rebuild")
            self.assertNotIn(1, result_ids, "id=1 must remain filtered after rebuild")
            self.assertNotIn(2, result_ids, "id=2 must remain filtered after rebuild")

        # --- Tombstone file persists after rebuild+flush and survives reload ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            tombstone_path = index_path + ".tombstones.npy"
            index = cls(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1])
            index.rebuild(list(range(2, SIZE)))
            index.close()
            # Tombstone file must still exist — tombstones were preserved
            self.assertTrue(
                os.path.isfile(tombstone_path),
                "tombstone file must be written since tombstones are preserved after rebuild"
            )
            # Reload — deleted IDs must still be filtered
            reloaded = cls(index_file_path=index_path, top_k=TOP_K)
            self.assertEqual(len(reloaded._tombstones), 2)
            results = reloaded.search(data[0])
            result_ids = [r[1] for r in results]
            self.assertNotIn(0, result_ids, "id=0 must remain filtered after reload post-rebuild")
            self.assertNotIn(1, result_ids, "id=1 must remain filtered after reload post-rebuild")

        # --- Create via VectorBase factory ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = VectorBase(
                'faiss', top_k=3, dimension=DIM,
                index_path=index_path, index_type='hnsw_sq8'
            )
            data = np.random.randn(100, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(100))])
            self.assertEqual(index.search(data[0])[0][1], 0)

    def test_faiss_hnsw_pq(self):
        """Test HNSW+PQ and HNSW+PQ+Refine: training-buffer warmup, search,
        tombstone deletion, rebuild, persistence and the small-dataset lazy flush."""
        # Seed numpy for reproducible query DATA only. PQ k-means centroids are
        # seeded by faiss (not numpy) and trained multithreaded, so exact recall
        # on this tiny under-trained index (256 centroids from SIZE points) is
        # not deterministic. The assertions below check structural + re-rank
        # contracts, not self-retrieval, which PQ cannot guarantee here.
        np.random.seed(0)
        # m_pq must divide DIM (512). pq_train_size < SIZE so training fires
        # during the normal mul_add path.
        for index_type in ("hnsw_pq", "hnsw_pq_refine"):
            cls = partial(
                Faiss, dimension=DIM, index_type=index_type,
                m_pq=16, pq_train_size=512,
            )

            # --- Training-buffer warmup, add and search ---
            with TemporaryDirectory(dir='./') as root:
                index_path = str((Path(root) / 'index.bin').absolute())
                index = cls(index_file_path=index_path, top_k=TOP_K)
                data = np.random.randn(SIZE, DIM).astype(np.float32)
                index.mul_add(
                    [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
                )
                self.assertEqual(index.index_type, index_type)
                # SIZE > pq_train_size, so PQ trained and all vectors flushed.
                self.assertTrue(index._index.is_trained)
                self.assertEqual(index.count(), SIZE)
                # High per-call efSearch makes HNSW near-exhaustive over the PQ
                # codes. Assert the robust contracts: search returns TOP_K
                # distinct, valid ids.
                results = index.search(data[0], ef_search=SIZE)
                result_ids = [r[1] for r in results]
                self.assertEqual(len(results), TOP_K)
                self.assertEqual(len(set(result_ids)), TOP_K)
                self.assertTrue(all(0 <= i < SIZE for i in result_ids))
                if index_type == "hnsw_pq_refine":
                    # Refine re-ranks the PQ candidate shortlist by full-precision
                    # distance, so the returned ids must be ordered by true L2 —
                    # this is what refine adds over plain PQ, and it holds
                    # regardless of which candidates the lossy base surfaces.
                    true_d = [float(np.sum((data[0] - data[i]) ** 2)) for i in result_ids]
                    self.assertEqual(true_d, sorted(true_d))

            # --- Tombstone deletion filters results (shared hnsw path) ---
            with TemporaryDirectory(dir='./') as root:
                index_path = str((Path(root) / 'index.bin').absolute())
                index = cls(index_file_path=index_path, top_k=TOP_K)
                data = np.random.randn(SIZE, DIM).astype(np.float32)
                index.mul_add(
                    [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
                )
                index.delete([0])
                result_ids = [r[1] for r in index.search(data[0])]
                self.assertNotIn(0, result_ids)
                self.assertEqual(index.count(), SIZE)

            # --- Rebuild preserves tombstones (PQ inherits HNSW no-evict) ---
            with TemporaryDirectory(dir='./') as root:
                index_path = str((Path(root) / 'index.bin').absolute())
                index = cls(index_file_path=index_path, top_k=TOP_K)
                data = np.random.randn(SIZE, DIM).astype(np.float32)
                index.mul_add(
                    [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
                )
                index.delete([0, 1, 2])
                self.assertEqual(len(index._tombstones), 3)
                index.rebuild(list(range(3, SIZE)))
                self.assertEqual(len(index._tombstones), 3)
                result_ids = [r[1] for r in index.search(data[0])]
                for ghost in (0, 1, 2):
                    self.assertNotIn(ghost, result_ids)

            # --- Persistence: index + tombstones survive save/load ---
            with TemporaryDirectory(dir='./') as root:
                index_path = str((Path(root) / 'index.bin').absolute())
                index = cls(index_file_path=index_path, top_k=TOP_K)
                data = np.random.randn(SIZE, DIM).astype(np.float32)
                index.mul_add(
                    [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
                )
                index.delete([0, 1])
                index.close()
                reloaded = cls(index_file_path=index_path, top_k=TOP_K)
                self.assertEqual(len(reloaded._tombstones), 2)
                self.assertEqual(reloaded.count(), SIZE)
                result_ids = [r[1] for r in reloaded.search(data[0])]
                self.assertNotIn(0, result_ids)
                self.assertNotIn(1, result_ids)

        # --- Small dataset (< pq_train_size): lazy flush makes it searchable ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = Faiss(
                index_file_path=index_path, dimension=DIM, top_k=TOP_K,
                index_type="hnsw_pq", m_pq=16, pq_train_size=10_000,
            )
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            # Still buffered — training threshold not reached yet.
            self.assertFalse(index._index.is_trained)
            # search() must trigger the lazy train+flush so data is queryable.
            results = index.search(data[0])
            self.assertTrue(index._index.is_trained)
            self.assertEqual(len(results), TOP_K)
            self.assertEqual(index.count(), SIZE)

        # --- Create via VectorBase factory with PQ params ---
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = VectorBase(
                'faiss', top_k=3, dimension=DIM,
                index_path=index_path, index_type='hnsw_pq_refine',
                m_pq=16, k_factor=8, pq_train_size=256,
            )
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(SIZE))])
            # High efSearch so HNSW surfaces the self node for the refine re-rank.
            self.assertEqual(index.search(data[0], ef_search=SIZE)[0][1], 0)

    def test_hnswlib(self):
        cls = partial(Hnswlib, max_elements=MAX_ELEMENTS, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='hnswlib',
                top_k=3,
                dimension=DIM,
                index_path=index_path,
                max_elements=MAX_ELEMENTS,
            )

    def test_docarray(self):
        self._internal_test_normal(DocArrayIndex)
        self._internal_test_with_rebuild(DocArrayIndex)
        self._internal_test_reload(DocArrayIndex)
        self._internal_test_delete(DocArrayIndex)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='docarray', top_k=3, index_path=index_path
            )

    def _internal_test_normal(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_with_rebuild(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1, 2])
            index.rebuild(list(range(3, SIZE)))
            self.assertNotEqual(index.search(data[0])[0], 0)

    def _internal_test_reload(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.close()

            new_index = vector_class(
                index_file_path=index_path, top_k=TOP_K
            )
            self.assertEqual(len(new_index.search(data[0])), TOP_K)
            new_index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = new_index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_delete(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.delete([0, 1, 2, 3])
            self.assertNotEqual(index.search(data[0])[0][1], 0)
            if hasattr(index, 'count'):
                self.assertEqual(index.count(), 996)

    def _internal_test_create_from_vector_base(self, **kwargs):
        index = VectorBase(**kwargs)
        data = np.random.randn(100, DIM).astype(np.float32)
        index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(100))])
        self.assertEqual(index.search(data[0])[0][1], 0)
