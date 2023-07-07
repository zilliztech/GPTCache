import unittest

import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestUSearchDB(unittest.TestCase):
    def test_normal(self):
        size = 1000
        dim = 512
        top_k = 10
        weaviate = VectorBase(
            "weaviate",
            top_k = top_k
        )
        data = np.random.randn(size, dim).astype(np.float32)
        weaviate.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(size))])
        search_result = weaviate.search(data[0], top_k)
        self.assertEqual(len(search_result), top_k)
        weaviate.mul_add([VectorData(id=size, data=data[0])])
        ret = weaviate.search(data[0])
        self.assertIn(ret[0][1], [0, size])
        self.assertIn(ret[1][1], [0, size])
        weaviate.delete([0, 1, 2, 3, 4, 5, size])
        ret = weaviate.search(data[0])
        self.assertNotIn(ret[0][1], [0, size])
        weaviate.rebuild()
        weaviate.close()