
import unittest
import numpy as np
from gptcache.cache.vector_data import Chromadb


class TestChromadb(unittest.TestCase):
    def test_normal(self):
        db = Chromadb(**{'client_settings': {}, 'top_k': 3})
        for i in range(100):
            db.add(str(i), np.random.sample(10))
        self.assertEqual(len(db.search(np.random.sample(10))), 3)
        db.delete(['1', '3', '5', '7'])
        self.assertEqual(db._collection.count(), 96)
