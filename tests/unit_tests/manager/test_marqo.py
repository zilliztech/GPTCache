import unittest

import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestMarqo(unittest.TestCase):
    def test_normal(self):
        marqo_url = "http://0.0.0.0:8882"
        db = VectorBase("marqo", marqo_url=marqo_url, dimension=10, top_k=3)
        db.mul_add([VectorData(id=i, data=np.random.sample(10)) for i in range(100)])
        search_res = db.search(np.random.sample(10))
        self.assertEqual(len(search_res), 3)
        db.delete(["1", "10", "50", "70"])
        self.assertEqual(db._collection.count(), 96)
