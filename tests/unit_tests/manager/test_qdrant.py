import os
import unittest

import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestQdrant(unittest.TestCase):
    def test_normal(self):
        size = 10
        dim = 2
        top_k = 10
        qdrant = VectorBase(
            "qdrant",
            top_k=top_k,
            dimension=dim,
            location=":memory:"
        )
        data = np.random.randn(size, dim).astype(np.float32)
        qdrant.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(size))])
        search_result = qdrant.search(data[0], top_k)
        self.assertEqual(len(search_result), top_k)
        qdrant.mul_add([VectorData(id=size, data=data[0])])
        ret = qdrant.search(data[0])
        self.assertIn(ret[0][1], [0, size])
        self.assertIn(ret[1][1], [0, size])
        qdrant.delete([0, 1, 2, 3, 4, 5, size])
        ret = qdrant.search(data[0])
        self.assertNotIn(ret[0][1], [0, size])
        qdrant.rebuild()
        qdrant.close()
