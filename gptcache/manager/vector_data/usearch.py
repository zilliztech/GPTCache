import os
from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_usearch

import_usearch()

from usearch.index import Index  # pylint: disable=C0413
from usearch.compiled import MetricKind  # pylint: disable=C0413


class USearch(VectorBase):
    """vector store: Usearch

    :param index_path: the path to Usearch index, defaults to 'index.usearch'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param metric: the distance mrtric. 'l2', 'haversine' or other, default = 'ip'
    :type metric: str
    :param dtype: the quantization dtype, 'f16' or 'f8' if needed, default = 'f32'
    :type dtype: str
    :param connectivity: the frequency of the connections in the graph, optional
    :type connectivity: int
    :param expansion_add: the recall of indexing, optional
    :type expansion_add: int
    :param expansion_search: the quality of search, optional
    :type expansion_search: int
    """

    def __init__(
        self,
        index_file_path: str = "index.usearch",
        dimension: int = 64,
        top_k: int = 1,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._top_k = top_k
        self._index = Index(
            ndim=self._dimension,
            metric=getattr(MetricKind, metric.lower().capitalize()),
            dtype=dtype,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        if os.path.isfile(self._index_file_path):
            self._index.load(self._index_file_path)

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array, dtype=np.longlong)
        self._index.add(ids, np_data)

    def search(self, data: np.ndarray, top_k: int = -1):
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)
        ids, dist, _ = self._index.search(np_data, top_k)
        return list(zip(dist[0], ids[0]))

    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        raise NotImplementedError

    def flush(self):
        self._index.save(self._index_file_path)

    def close(self):
        self.flush()

    def count(self):
        return len(self._index)
