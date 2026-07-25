import os
import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from gptcache.manager import get_data_manager, CacheBase, VectorBase

DIM = 8


def mock_embeddings():
    return np.random.random((DIM,)).astype("float32")


class TestEviction(unittest.TestCase):
    """Test data eviction"""

    def test_eviction_lru(self):
        with TemporaryDirectory(dir='./') as root:
            db_path = Path(root) / 'sqlite.db'
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            vector_base = VectorBase("faiss", dimension=DIM)
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2, eviction="LRU"
            )
            for i in range(19):
                question = f"foo{i}"
                answer = f"receiver the foo {i}"
                data_manager.save(question, answer, mock_embeddings())
            cache_count = data_manager.s.count()
            self.assertEqual(cache_count, 9)
            ids = data_manager.s.get_ids(deleted=True)
            self.assertEqual(len(ids), 0)

    def test_eviction_fifo(self):
        with TemporaryDirectory(dir='./') as root:
            db_path = Path(root) / 'sqlite.db'
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            vector_base = VectorBase("faiss", dimension=DIM)
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2, eviction="FIFO"
            )
            for i in range(18):
                question = f"foo{i}"
                answer = f"receiver the foo {i}"
                data_manager.save(question, answer, mock_embeddings())

            cache_count = data_manager.s.count()
            self.assertEqual(cache_count, 10)

    def test_eviction_gdsf_keeps_expensive_answers(self):
        """End-to-end check that the data manager feeds answer costs to GDSF.

        We insert one very long (expensive) answer first, then flood the
        cache with short answers. Under LRU the oldest entry would be evicted
        first; under GDSF the expensive entry must survive because its
        cost-based priority is far above the short answers'.
        """
        with TemporaryDirectory(dir='./') as root:
            db_path = Path(root) / 'sqlite.db'
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            vector_base = VectorBase("faiss", dimension=DIM)
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2, eviction="GDSF"
            )
            data_manager.save("expensive", "x" * 5000, mock_embeddings())
            for i in range(18):
                data_manager.save(f"foo{i}", f"short answer {i}", mock_embeddings())
            ids = data_manager.s.get_ids(deleted=False)
            self.assertIn(1, [int(i) for i in ids])
            self.assertLessEqual(len(ids), 10)

    # def test_eviction_milvus(self):
    #     cache_base = CacheBase('sqlite', sql_url='sqlite:///./gptcache2.db')
    #     vector_base = VectorBase('milvus', dimension=DIM, host='172.16.70.4', collection_name='gptcache2')
    #     data_manager = get_data_manager(cache_base, vector_base, max_size=10, clean_size=2, eviction='LRU')
    #     for i in range(10):
    #         question = f'foo{i}'
    #         answer = f'receiver the foo {i}'
    #         data_manager.save(question, answer, mock_embeddings())
    #
    #     cache_count = data_manager.s.count(is_all=True)
    #     self.assertEqual(cache_count, 10)
