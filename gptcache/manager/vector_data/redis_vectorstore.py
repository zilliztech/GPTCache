from typing import List

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils.log import gptcache_log
from gptcache.utils import import_docarray
import numpy as np
import redis

from redis.client import Redis as RedisType
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class RedisVectorStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        dimension: int = 0,
        collection_name: str = "gptcache",
        top_k: int = 1,
        doc_prefix: str = "doc",
    ):
        self._client = redis.Redis(host=host, port=port)
        self.top_k = top_k
        self.dimension = dimension
        self.doc_prefix = doc_prefix
        self.collection_name = collection_name

        self._create_collection(collection_name)

    def _check_index_exists(client: RedisType, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            client.ft(index_name).info()
        except:  # noqa: E722
            gptcache_log.info("Index does not exist")
            return False
        gptcache_log.info("Index already exists")
        return True

    def _create_collection(self, collection_name):
        if not self._check_index_exists(collection_name):
            schema = (
                TagField("tag"),  # Tag Field Name
                VectorField(
                    "vector",  # Vector Field Name
                    "FLAT",
                    {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": self.dimension,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[self.doc_prefix], index_type=IndexType.HASH
            )

            # create Index
            self._client.ft(collection_name).create_index(
                fields=schema, definition=definition
            )
        else:
            gptcache_log.info(
                f"The {collection_name} already exists, and it will be used directly"
            )

    def mul_add(self, datas: List[VectorData]):
        pass

    def search(self, data: np.ndarray, top_k: int):
        query = (
            Query("*=>[KNN 2 @vector $vec as score]")
            .sort_by("score")
            .return_fields("id", "score")
            .paging(0, 2)
            .dialect(2)
        )
        query_params = {"vec": np.random.rand(self.dimension).astype(np.float32).tobytes()}
        results = self._client.ft(self.collection_name).search(query, query_params).docs
        return [(result.score, result.id) for result in results]

    def rebuild(self, ids=None) -> bool:
        pass

    def delete(self, ids) -> bool:
        pass
