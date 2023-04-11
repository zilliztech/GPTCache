import os
import numpy as np

from gptcache.manager.vector_data.base import VectorBase, ClearStrategy
from gptcache.utils import import_hnswlib

import_hnswlib()

import hnswlib  # pylint: disable=C0413


class Hnswlib(VectorBase):
    """vector store: hnswlib"""

    def __init__(self, index_file_path: int, top_k: str, dimension: int, max_elements: int):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._max_elements = max_elements
        self._index =  hnswlib.Index(space="l2", dim=self._dimension)
        self._top_k = top_k
        if os.path.isfile(self._index_file_path):
            self._index.load_index(self._index_file_path, max_elements=max_elements)
        else:
            self._index.init_index(max_elements=max_elements, ef_construction=100, M=16)
            self._index.set_ef(self._top_k * 2)

    def add(self, key: int, data: "ndarray"):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        self._index.add_items(np_data, np.asarray([key]))

    def _mult_add(self, data, keys):
        np_data = np.array(data).astype("float32")
        self._index.add_items(np_data, np.asarray(keys))

    def search(self, data: "ndarray"):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        ids, dist = self._index.knn_query(data=np_data, k=self._top_k)
        return list(zip(dist[0], ids[0]))

    def clear_strategy(self):
        return ClearStrategy.REBUILD

    def rebuild(self, all_data, keys):
        new_index = hnswlib.Index(space="l2", dim=self._dimension)
        new_index.init_index(max_elements=self._max_elements, ef_construction=100, M=16)
        new_index.set_ef(self._top_k * 2)
        self._index = new_index
        self._mult_add(all_data, keys)

    def close(self):
        self._index.save_index(self._index_file_path)
