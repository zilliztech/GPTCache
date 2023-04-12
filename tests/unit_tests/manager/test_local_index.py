import unittest
import numpy as np
from functools import partial
from tempfile import TemporaryDirectory
from pathlib import Path

from gptcache.manager.vector_data.faiss import Faiss
from gptcache.manager.vector_data.hnswlib_store import Hnswlib
from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData


class TestLocalIndex(unittest.TestCase):
    def test_faiss(self):
        self._internal_test_normal(Faiss)
        self._internal_test_with_rebuild(Faiss)
        self._internal_test_reload(Faiss)
        self._internal_test_delete(Faiss)
        
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(name='faiss', top_k=3, dimension=512, index_path=index_path)

    def test_hnswlib(self):
        cls = partial(Hnswlib, max_elements=10000)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(name='hnswlib', top_k=3, dimension=512, index_path=index_path, max_elements=10000)

    def _internal_test_normal(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = vector_class(index_path, dim, top_k)
            data = np.random.randn(size, dim).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(size)))])
            self.assertEqual(len(index.search(data[0])), top_k)
            index.mul_add([VectorData(id=size, data=data[0])])
            ret = index.search(data[0])
            self.assertIn(ret[0][1], [0, size])
            self.assertIn(ret[1][1], [0, size])

    def _internal_test_with_rebuild(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = vector_class(index_path, dim, top_k)
            data = np.random.randn(size, dim).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(size)))])
            index.delete([0, 1, 2])
            index.rebuild(list(range(3, size)))
            self.assertNotEqual(index.search(data[0])[0], 0)

    def _internal_test_reload(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = vector_class(index_path, dim, top_k)
            data = np.random.randn(size, dim).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(size)))])
            index.close()

            new_index = vector_class(index_path, dim, top_k)
            self.assertEqual(len(new_index.search(data[0])), top_k)
            new_index.mul_add([VectorData(id=size, data=data[0])])            
            ret = new_index.search(data[0])
            self.assertIn(ret[0][1], [0, size])
            self.assertIn(ret[1][1], [0, size])

    def _internal_test_delete(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            size = 1000
            dim = 512
            top_k = 10
            index = vector_class(index_path, dim, top_k)
            data = np.random.randn(size, dim).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(size)))])
            self.assertEqual(len(index.search(data[0])), top_k)
            index.delete([0, 1, 2, 3])
            self.assertNotEqual(index.search(data[0])[0][1], 0)
            if hasattr(index, 'count'):
                self.assertEqual(index.count(), 996)

    def _internal_test_create_from_vector_base(self, **kwargs):
        index = VectorBase(**kwargs)
        data = np.random.randn(100, 512).astype(np.float32)
        index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(100))])
        self.assertEqual(index.search(data[0])[0][1], 0)
