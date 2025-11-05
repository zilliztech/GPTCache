from typing import List, Optional

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_redis
from gptcache.utils.log import gptcache_log

import_redis()

# pylint: disable=C0413
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, VectorField
from redis.client import Redis
from redis.exceptions import ResponseError


class RedisVectorStore(VectorBase):
    """ vector store: Redis or Valkey

    :param host: redis host, defaults to "localhost".
    :type host: str
    :param port: redis port, defaults to "6379".
    :type port: str
    :param username: redis username, defaults to "".
    :type username: str
    :param password: redis password, defaults to "".
    :type password: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param collection_name: the name of the index for Redis, defaults to "gptcache".
    :type collection_name: str
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int

    Example:
        .. code-block:: python

            from gptcache.manager import VectorBase

            vector_base = VectorBase("redis", dimension=10)
    """
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        dimension: int = 0,
        collection_name: str = "gptcache",
        top_k: int = 1,
        namespace: str = "",
    ):
        self._client = Redis(
            host=host, port=int(port), username=username, password=password
        )
        self.top_k = top_k
        self.dimension = dimension
        self.collection_name = collection_name
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"  # Prefix with the specified namespace
        self._create_collection(collection_name)

    _sortby_supported: bool | None = None

    def _check_sortby_support(self, index_name: str, sort_field: str) -> bool:
        """
        Runtime capability detection for FT.SEARCH SORTBY.
        Issues a zero-result query using SORTBY against the provided index.
        - If the server rejects the keyword (e.g., Valkey builds without SORTBY), we cache False and return False.
        - If the server *parses* SORTBY but complains about the field or schema (e.g., 'Property is not sortable'
        or 'No such field'), we treat that as SORTBY being *supported* and bubble up the original error when the real
        query runs.
        """

        if self._sortby_supported is not None:
            return self._sortby_supported

        try:
            self._client.execute_command("FT.SEARCH", index_name, "*", "SORTBY", sort_field, "ASC", "LIMIT", 0, 0)
            self._sortby_supported = True

        except ResponseError as e:
            if "SORTBY" in str(e):
                self._sortby_supported = False
            else:
                self._sortby_supported = True
            return self._sortby_supported

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except:  # pylint: disable=W0702
            gptcache_log.info("Index does not exist")
            return False
        gptcache_log.info("Index already exists")
        return True

    def _create_collection(self, collection_name):
        if self._check_index_exists(collection_name):
            gptcache_log.info(
                "The %s already exists, and it will be used directly", collection_name
            )
        else:
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

    def mul_add(self, datas: List[VectorData], **_):
        pipe = self._client.pipeline()

        for data in datas:
            key: int = data.id
            obj = {
                "vector": data.data.astype(np.float32).tobytes(),
            }
            pipe.hset(f"{self.doc_prefix}{key}", mapping=obj)

        pipe.execute()

    def search(self, data: np.ndarray, top_k: int = -1, **_):
        query = (
            Query(
                f"*=>[KNN {top_k if top_k > 0 else self.top_k} @vector $vec as score]"
            )
            .return_fields("id", "score")
            .paging(0, top_k if top_k > 0 else self.top_k)
            .dialect(2)
        )
        if self._check_sortby_support(self.collection_name, "score"):
            query = query.sort_by("score")

        query_params = {"vec": data.astype(np.float32).tobytes()}
        results = (
            self._client.ft(self.collection_name)
            .search(query, query_params=query_params)
            .docs
        )
        return [(float(result.score), int(result.id[len(self.doc_prefix):])) for result in results]

    def rebuild(self, ids=None) -> bool:
        pass

    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
        pipe.execute()
