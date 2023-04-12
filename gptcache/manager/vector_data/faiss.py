import os
from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, ClearStrategy, VectorData
from gptcache.utils import import_faiss

import_faiss()

import faiss  # pylint: disable=C0413
from faiss import Index  # pylint: disable=C0413


class Faiss(VectorBase):
    """vector store: Faiss"""

    index: Index

    def __init__(self, index_file_path, dimension, top_k, skip_file=False):
        self.index_file_path = index_file_path
        self.dimension = dimension
        self.index = faiss.index_factory(self.dimension, "IDMap,Flat", faiss.METRIC_L2)
        self.top_k = top_k
        if os.path.isfile(index_file_path) and not skip_file:
            self.index = faiss.read_index(index_file_path)

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        self.index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray):
        if self.index.ntotal == 0:
            return None
        np_data = np.array(data).astype("float32").reshape(1, -1)
        dist, ids = self.index.search(np_data, self.top_k)
        ids = [int(i) for i in ids[0]]
        return zip(dist[0], ids)

    def clear_strategy(self):
        return ClearStrategy.REBUILD

    def rebuild(self, all_data, keys):
        self.index = faiss.index_factory(self.dimension, "IDMap,Flat", faiss.METRIC_L2)
        datas = []
        for i, key in enumerate(keys):
            datas.append(VectorData(id=key, data=all_data[i]))
        self.mul_add(datas)

    def close(self):
        faiss.write_index(self.index, self.index_file_path)
