from gptcache.util import import_qdrant_client
import_qdrant_client()

import qdrant_client
from qdrant_client.http import models
from .base import VectorBase, ClearStrategy


class Qdrant(VectorBase):
    def __init__(self, **kwargs):
        self._client = kwargs.get("client", None)
        self._top_k = kwargs.get("top_k", 1)
        self._distance = kwargs.get("distance", models.Distance.COSINE)
        if self._client is None:
            self._client = qdrant_client.QdrantClient(":memory:")

        self._collection_name = kwargs.get("collection_name", "gptcache")
        self._dim = kwargs.get("dim", 0)
        if self._dim <= 0:
            raise ValueError(f"dim shuold > 0")

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=self._dim, distance=self._distance)
        )

    def add(self, key, data):
        self._client.upsert(
            collection_name = self._collection_name,
            points=[
                models.PointStruct(
                    id=key,
                    vector=data.tolist(),
                ),
            ]
        )

    def search(self, data):
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=data,
            with_payload=False,
            with_vectors=True,
            limit=self._top_k,
        )
        return [(item.score, item.vector) for item in results]

    def delete(self, ids):
        self._client.delete(self._collection_name, ids)

    def close(self):
        return True

    def clear_strategy(self):
        return ClearStrategy.DELETE
