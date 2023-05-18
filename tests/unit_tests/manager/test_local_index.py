import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from gptcache.manager.vector_data import VectorBase
from gptcache.manager.vector_data.base import VectorData
from gptcache.manager.vector_data.docarray_index import DocArrayIndex
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.manager.vector_data.hnswlib_store import Hnswlib

DIM = 512
MAX_ELEMENTS = 10000
SIZE = 1000
TOP_K = 10


class TestLocalIndex(unittest.TestCase):
    def test_faiss(self):
        cls = partial(Faiss, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='faiss', top_k=3, dimension=DIM, index_path=index_path
            )

    @pytest.mark.tags("L2")
    def test_hnswlib(self):
        cls = partial(Hnswlib, max_elements=MAX_ELEMENTS, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='hnswlib',
                top_k=3,
                dimension=DIM,
                index_path=index_path,
                max_elements=MAX_ELEMENTS,
            )

    def test_docarray(self):
        self._internal_test_normal(DocArrayIndex)
        self._internal_test_with_rebuild(DocArrayIndex)
        self._internal_test_reload(DocArrayIndex)
        self._internal_test_delete(DocArrayIndex)

        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            self._internal_test_create_from_vector_base(
                name='docarray', top_k=3, index_path=index_path
            )

    def _internal_test_normal(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_with_rebuild(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.delete([0, 1, 2])
            index.rebuild(list(range(3, SIZE)))
            self.assertNotEqual(index.search(data[0])[0], 0)

    def _internal_test_reload(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            index.close()

            new_index = vector_class(
                index_file_path=index_path, top_k=TOP_K
            )
            self.assertEqual(len(new_index.search(data[0])), TOP_K)
            new_index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = new_index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_delete(self, vector_class):
        with TemporaryDirectory(dir='./') as root:
            index_path = str((Path(root) / 'index.bin').absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add(
                [VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))]
            )
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.delete([0, 1, 2, 3])
            self.assertNotEqual(index.search(data[0])[0][1], 0)
            if hasattr(index, 'count'):
                self.assertEqual(index.count(), 996)

    def _internal_test_create_from_vector_base(self, **kwargs):
        index = VectorBase(**kwargs)
        data = np.random.randn(100, DIM).astype(np.float32)
        index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(100))])
        self.assertEqual(index.search(data[0])[0][1], 0)
