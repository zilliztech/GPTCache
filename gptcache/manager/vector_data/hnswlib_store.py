import os
from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, ClearStrategy, VectorData
from gptcache.utils import import_hnswlib

import_hnswlib()

import hnswlib  # pylint: disable=C0413


class Hnswlib(VectorBase):
    """vector store: hnswlib"""

    def __init__(self, index_file_path: str, top_k: int, dimension: int, max_elements: int):
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

    def add(self, key: int, data: np.ndarray):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        self._index.add_items(np_data, np.array([key]))

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        self._index.add_items(np_data, ids)

    def search(self, data: np.ndarray):
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
        datas = []
        for i, key in enumerate(keys):
            datas.append(VectorData(id=key, data=all_data[i]))
        self.mul_add(datas)

    def close(self):
        self._index.save_index(self._index_file_path)
