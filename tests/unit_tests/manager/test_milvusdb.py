import unittest
import numpy as np
from tempfile import TemporaryDirectory

from gptcache.manager.vector_data.milvus import Milvus
from gptcache.manager.vector_data import VectorBase


class TestMilvusDB(unittest.TestCase):
    def test_normal(self):
        with TemporaryDirectory(dir='./') as root:
            size = 1000
            dim = 512
            top_k = 10
            
            db = Milvus(top_k=top_k, dimension=dim, port='10086', local_mode=True, local_data=str(root),
                        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
            data = np.random.randn(size, dim).astype(np.float32)
            for i in range(size):
                db.add(i, data[i])
            self.assertEqual(len(db.search(data[0])), top_k)
            db.add(size, data[0])
            ret = db.search(data[0])
            self.assertIn(ret[0][1], [0, size])
            self.assertIn(ret[1][1], [0, size])
            db.delete([0, 1, 2, 3, 4, 5, size])
            ret = db.search(data[0])
            self.assertNotIn(ret[0][1], [0, size])
            db.rebuild()
            db.close()
