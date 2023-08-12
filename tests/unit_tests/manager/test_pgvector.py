import os
import unittest

import numpy as np

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestPgvector(unittest.TestCase):
    def test_normal(self):
        size = 1000
        dim = 10
        top_k = 10

        url = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

        db = VectorBase(
            "pgvector",
            top_k=top_k,
            dimension=dim,
            url=url,
            index_params={
                "index_type": "L2",
                "params": {"lists": 100, "probes": 10},
            },
        )
        db.delete([i for i in range(size)])
        data = np.random.randn(size, dim).astype(np.float32)
        db.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(size))])
        self.assertEqual(len(db.search(data[0])), top_k)
        db.mul_add([VectorData(id=size, data=data[0])])
        ret = db.search(data[0])
        print(ret)
        self.assertIn(ret[0][0], [0, size])
        self.assertIn(ret[1][0], [0, size])
        db.delete([0, 1, 2, 3, 4, 5, size])
        ret = db.search(data[0])
        print(ret)
        self.assertNotIn(ret[0][0], [0, size])
        db.rebuild()
        db.close()
