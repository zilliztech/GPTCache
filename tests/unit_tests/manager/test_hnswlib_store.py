import unittest
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path

from gptcache.manager.vector_data.hnswlib_store import Hnswlib
from gptcache.manager.vector_data.base import ClearStrategy
from gptcache.manager.vector_data import VectorBase


class TestHnswlib(unittest.TestCase):
    def test_normal(self):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'nmslib.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = Hnswlib(index_path, top_k, dim, size + 10)
            data = np.random.randn(size, dim).astype(np.float32)
            index._mult_add(data, list(range(size)))
            self.assertEqual(len(index.search(data[0])), top_k)
            index.add(size, data[0])
            ret = index.search(data[0])
            self.assertEqual(ret[0][1], 0)
            self.assertEqual(ret[1][1], size)

    def test_with_rebuild(self):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'nmslib.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = Hnswlib(index_path, top_k, dim, size + 10)
            data = np.random.randn(size, dim).astype(np.float32)
            index._mult_add(data, list(range(1, data.shape[0] + 1)))

            self.assertEqual(index.clear_strategy(), ClearStrategy.REBUILD)
            index.rebuild(data[1:], list(range(size - 1)))
            self.assertNotEqual(index.search(data[0])[0], 0)

    def test_reload(self):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'nmslib.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = Hnswlib(index_path, top_k, dim, size + 10)
            data = np.random.randn(size, dim).astype(np.float32)
            index._mult_add(data, list(range(size)))
            index.close()

            new_index = Hnswlib(index_path, top_k, dim, size + 10)
            self.assertEqual(len(new_index.search(data[0])), top_k)
            new_index.add(size, data[0])
            ret = new_index.search(data[0])
            self.assertEqual(ret[0][1], 0)
            self.assertEqual(ret[1][1], size)

    def test_create_from_vector_base(self):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'nmslib.bin').absolute())
            index = VectorBase('hnswlib', top_k=3, dimension=512,
                               max_elements=5000, index_path=index_path)
            data = np.random.randn(100, 512).astype(np.float32)
            for i in range(100):
                index.add(i, data[i])
            self.assertEqual(index.search(data[0])[0][1], 0)
