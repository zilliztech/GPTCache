import unittest
import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestWeaviateDB(unittest.TestCase):
    def test_normal(self):
        size = 1000
        dim = 512
        top_k = 10

        db = VectorBase(
            "weaviate",
            top_k=top_k
        )

        db._create_class()
        data = np.random.randn(size, dim).astype(np.float32)
        db.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(size))])
        self.assertEqual(len(db.search(data[0])), top_k)
        db.mul_add([VectorData(id=size, data=data[0])])
        ret = db.search(data[0])
        self.assertIn(ret[0][1], [0, size])
        self.assertIn(ret[1][1], [0, size])
        db.delete([0, 1, 2, 3, 4, 5, size])
        ret = db.search(data[0])
        self.assertNotIn(ret[0][1], [0, size])
        db.rebuild()
        db.update_embeddings(6, data[7])
        emb = db.get_embeddings(6)
        self.assertEqual(emb.tolist(), data[7].tolist())
        emb = db.get_embeddings(0)
        self.assertIsNone(emb)
        db.close()
