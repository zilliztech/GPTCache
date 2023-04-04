
import unittest
import numpy as np
from gptcache.cache.vector_data.qdrant import Qdrant
import qdrant_client


class TestChromadb(unittest.TestCase):
    def test_normal(self):
        db = Qdrant(**{'top_k': 4, 'dim': 10})
        self.assertEqual(db._client.count(db._collection_name).count, 0)
        for i in range(100):
            db.add(i, np.random.sample(10))
        self.assertEqual(db._client.count(db._collection_name).count, 100)
        self.assertEqual(len(db.search(np.random.sample(10))), 4)
        db.delete([1, 3, 5, 7])
        self.assertEqual(db._client.count(db._collection_name).count, 96)
