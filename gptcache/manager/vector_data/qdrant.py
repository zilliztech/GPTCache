from typing import List, Optional
import numpy as np

from gptcache.utils import import_qdrant
from gptcache.utils.log import gptcache_log
from gptcache.manager.vector_data.base import VectorBase, VectorData

import_qdrant()

from qdrant_client import QdrantClient  # pylint: disable=C0413
from qdrant_client.conversions import common_types as types  # pylint: disable=C0413
from qdrant_client.models import PointStruct  # pylint: disable=C0413


class QdrantVectorStore(VectorBase):

    def __init__(
            self,
            url: Optional[str] = None,
            port: Optional[int] = 6333,
            grpc_port: int = 6334,
            prefer_grpc: bool = False,
            https: Optional[bool] = None,
            api_key: Optional[str] = None,
            prefix: Optional[str] = None,
            timeout: Optional[float] = None,
            host: Optional[str] = None,
            collection_name: Optional[str] = "gptcache",
            location: Optional[str] = "./qdrant",
            dimension: int = 0,
            top_k: int = 1,
            flush_interval_sec: int = 5,
            index_params: Optional[dict] = None,
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Qdrant vector store."
            )
        self._client: QdrantClient
        self._collection_name = collection_name
        self._in_memory = location == ":memory:"
        self._closeable = self._in_memory or location is not None
        self.dimension = dimension
        self.top_k = top_k
        if self._in_memory or location is not None:
            self._create_local(location)
        else:
            self._create_remote(url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https)
        self._create_collection(collection_name, flush_interval_sec, index_params)

    def _create_local(self, location):
        self._client = QdrantClient(location=location)

    def _create_remote(self, url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https):
        self._client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            timeout=timeout,
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            prefix=prefix,
            https=https,
        )

    def _create_collection(self, collection_name: str, flush_interval_sec: int, index_params: Optional[dict] = None):
        hnsw_config = types.HnswConfigDiff(**(index_params or {}))
        vectors_config = types.VectorParams(size=self.dimension, distance=types.Distance.COSINE,
                                            hnsw_config=hnsw_config)
        optimizers_config = types.OptimizersConfigDiff(deleted_threshold=0.2, vacuum_min_vector_number=1000,
                                                       flush_interval_sec=flush_interval_sec)
        # check if the collection exists
        existing_collection = self._client.get_collection(collection_name=collection_name)
        if existing_collection:
            gptcache_log.warning("The %s collection already exists, and it will be used directly.", collection_name)
            self.col = existing_collection
        else:
            self.col = self._client.create_collection(collection_name=collection_name, vectors_config=vectors_config,
                                                      optimizers_config=optimizers_config)

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        entities = [id_array, np_data]
        points = [PointStruct(id=_id, vector=vector) for _id, vector in zip(*entities)]
        self._client.upsert(collection_name=self._collection_name, points=points, wait=False)

    def search(self, data: np.ndarray, top_k: int = -1):
        if top_k == -1:
            top_k = self.top_k
        reshaped_data = data.reshape(1, -1).tolist()
        search_result = self._client.search(collection_name=self._collection_name, query_vector=reshaped_data,
                                            limit=top_k)
        return list(map(lambda x: (x.id, x.score), search_result))

    def delete(self, ids: List[str]):
        self._client.delete_vectors(collection_name=self._collection_name, vectors=ids)

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        optimizers_config = types.OptimizersConfigDiff(deleted_threshold=0.2, vacuum_min_vector_number=1000)
        self._client.update_collection(collection_name=self._collection_name, optimizer_config=optimizers_config)

    def flush(self):
        # no need to flush manually as qdrant flushes automatically based on the optimizers_config for remote Qdrant
        if self._closeable:
            self._client._save()  # pylint: disable=protected-access

    def close(self):
        self.flush()
