import os
import unittest
import numpy as np

from gptcache.manager import get_data_manager, CacheBase, VectorBase

DIM = 8


def mock_embeddings():
    return np.random.random((DIM,)).astype("float32")


class TestEviction(unittest.TestCase):
    """Test data eviction"""

    def test_eviction_lru(self):
        cache_base = CacheBase("sqlite", sql_url="sqlite:///./gptcache0.db")
        vector_base = VectorBase("faiss", dimension=DIM)
        data_manager = get_data_manager(
            cache_base, vector_base, max_size=10, clean_size=2, eviction="LRU"
        )
        for i in range(15):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            data_manager.save(question, answer, mock_embeddings())
        cache_count = data_manager.s.count()
        self.assertEqual(cache_count, 9)

    def test_eviction_fifo(self):
        cache_base = CacheBase("sqlite", sql_url="sqlite:///./gptcache1.db")
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

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove("gptcache0.db")
        os.remove("gptcache1.db")
        # os.remove('gptcache2.db')
