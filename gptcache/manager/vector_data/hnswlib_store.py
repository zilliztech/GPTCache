import os
from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_hnswlib

import_hnswlib()

import hnswlib  # pylint: disable=C0413


class Hnswlib(VectorBase):
    """vector store: hnswlib

    :param index_path: the path to hnswlib index, defaults to 'hnswlib_index.bin'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param max_elements: max_elements of hnswlib, defaults 100000.
    :type max_elements: int
    """

    def __init__(self, index_file_path: str, dimension: int, top_k: int, max_elements: int):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._max_elements = max_elements
        self._index = hnswlib.Index(space="l2", dim=self._dimension)
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

    def search(self, data: np.ndarray, top_k: int = -1):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        if top_k == -1:
            top_k = self._top_k
        ids, dist = self._index.knn_query(data=np_data, k=top_k)
        return list(zip(dist[0], ids[0]))

    def rebuild(self, ids):
        all_data = self._index.get_items(ids)
        new_index = hnswlib.Index(space="l2", dim=self._dimension)
        new_index.init_index(max_elements=self._max_elements, ef_construction=100, M=16)
        new_index.set_ef(self._top_k * 2)
        self._index = new_index
        datas = []
        for key, data in zip(ids, all_data):
            datas.append(VectorData(id=key, data=data))
        self.mul_add(datas)

    def delete(self, ids):
        for i in ids:
            self._index.mark_deleted(i)

    def flush(self):
        self._index.save_index(self._index_file_path)

    def close(self):
        self.flush()
