import os
import unittest
import numpy as np

from gptcache.manager.factory import get_data_manager

DIM = 8


def mock_embeddings():
    return np.random.random((DIM, )).astype('float32')


class TestEviction(unittest.TestCase):
    def test_eviction_lru(self):
        url = 'sqlite:///./gptcache0.db'
        data_manager = get_data_manager('sqlite', 'faiss', dimension=DIM, max_size=10, clean_size=2, eviction='LRU', sql_url=url)
        for i in range(15):
            question = f'foo{i}'
            answer = f'receiver the foo {i}'
            data_manager.save(question, answer, mock_embeddings())
        cache_count = data_manager.s.count()
        self.assertEqual(cache_count, 9)

    def test_eviction_fifo(self):
        url = 'sqlite:///./gptcache1.db'
        data_manager = get_data_manager('sqlite', 'faiss', dimension=DIM, max_size=10, clean_size=2, eviction='FIFO', sql_url=url)
        for i in range(18):
            question = f'foo{i}'
            answer = f'receiver the foo {i}'
            data_manager.save(question, answer, mock_embeddings())

        cache_count = data_manager.s.count()
        self.assertEqual(cache_count, 10)

    # def test_eviction_milvus(self):
    #     url = 'sqlite:///./gptcache2.db'
    #     data_manager = get_data_manager('sqlite', 'milvus', dimension=DIM, max_size=10, clean_size=2, eviction='FIFO', sql_url=url)
    #     for i in range(10):
    #         question = f'foo{i}'
    #         answer = f'receiver the foo {i}'
    #         data_manager.save(question, answer, mock_embeddings())
    #
    #     cache_count = data_manager.s.count(is_all=True)
    #     self.assertEqual(cache_count, 10)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('gptcache0.db')
        os.remove('gptcache1.db')
        # os.remove('gptcache2.db')
