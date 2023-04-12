import os
import numpy as np

from gptcache.manager.vector_data.base import VectorBase
from gptcache.utils import import_faiss

import_faiss()

import faiss  # pylint: disable=C0413
from faiss import Index  # pylint: disable=C0413


class Faiss(VectorBase):
    """vector store: Faiss"""

    index: Index

    def __init__(self, index_file_path, dimension, top_k):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._index = faiss.index_factory(self._dimension, "IDMap,Flat", faiss.METRIC_L2)
        self._top_k = top_k
        if os.path.isfile(index_file_path):
            self._index = faiss.read_index(index_file_path)

    def add(self, key: int, data: "ndarray"):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        ids = np.array([key])
        self._index.add_with_ids(np_data, ids)

    def _mult_add(self, data, keys):
        np_data = np.array(data).astype("float32")
        ids = np.array(keys).astype(np.int64)
        self._index.add_with_ids(np_data, ids)

    def search(self, data: "ndarray"):
        if self._index.ntotal == 0:
            return None
        np_data = np.array(data).astype("float32").reshape(1, -1)
        dist, ids = self._index.search(np_data, self._top_k)
        ids = [int(i) for i in ids[0]]
        return list(zip(dist[0], ids))

    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        ids_to_remove = np.array(ids)
        self._index.remove_ids(faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove)))

    def close(self):
        faiss.write_index(self._index, self._index_file_path)

    def count(self):
        return self._index.ntotal
