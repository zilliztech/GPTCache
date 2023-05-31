import unittest
import numpy as np

from gptcache.manager.vector_data.usearch import USearch
from gptcache.manager.vector_data.base import VectorData


class TestUSearchDB(unittest.TestCase):
    def test_normal(self):
        size = 1000
        dim = 512
        top_k = 10

        db = USearch(
            index_file_path='./index.usearch',
            dimension=dim,
            top_k=top_k,
            metric='cos',
            dtype='f32',
        )
        db.mul_add([VectorData(id=i, data=np.random.rand(dim))
                   for i in range(size)])
        self.assertEqual(len(db.search(np.random.rand(dim))), top_k)
        self.assertEqual(db.count(), size)
        db.close()
