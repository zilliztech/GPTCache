import unittest

import numpy as np

from gptcache.manager import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestChromadb(unittest.TestCase):
    def test_normal(self):
        db = VectorBase("chromadb", client_settings={}, top_k=3)
        db.mul_add([VectorData(id=i, data=np.random.sample(10)) for i in range(100)])
        search_res = db.search(np.random.sample(10))
        self.assertEqual(len(search_res), 3)
        db.delete(["1", "3", "5", "7"])
        self.assertEqual(db._collection.count(), 96)
