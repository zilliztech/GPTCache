import os
import numpy as np

from gptcache.manager.vector_data.base import VectorBase, ClearStrategy
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

    def add(self, key: int, data: "ndarray"):
        np_data = np.array(data).astype("float32").reshape(1, -1)
        ids = np.array([key])
        self.index.add_with_ids(np_data, ids)

    def _mult_add(self, datas, keys):
        np_data = np.array(datas).astype("float32")
        ids = np.array(keys).astype(np.int64)
        self.index.add_with_ids(np_data, ids)

    def search(self, data: "ndarray"):
        if self.index.ntotal == 0:
            return None
        np_data = np.array(data).astype("float32").reshape(1, -1)
        dist, ids = self.index.search(np_data, self.top_k)
        ids = [int(i) for i in ids[0]]
        return zip(dist[0], ids)

    def clear_strategy(self):
        return ClearStrategy.REBUILD

    def rebuild(self, all_data, keys):
        f = Faiss(
            self.index_file_path, self.dimension, top_k=self.top_k, skip_file=True
        )
        f._mult_add(all_data, keys)  # pylint: disable=protected-access
        return f

    def close(self):
        faiss.write_index(self.index, self.index_file_path)
