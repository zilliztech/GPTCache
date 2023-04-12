import unittest
import numpy as np

from gptcache.manager.vector_data.base import VectorData
from gptcache.manager.vector_data.chroma import Chromadb


class TestChromadb(unittest.TestCase):
    def test_normal(self):
        db = Chromadb(**{"client_settings": {}, "top_k": 3})
        db.mul_add([VectorData(id=i, data=np.random.sample(10)) for i in range(100)])
        self.assertEqual(len(db.search(np.random.sample(10))), 3)
        db.delete(["1", "3", "5", "7"])
        self.assertEqual(db._collection.count(), 96)
