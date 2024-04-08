from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils.log import gptcache_log
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError


class ElasticSearchStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "9200",
        username: str = "",
        password: str = "",
        dimension: int = 0,
        collection_name: str = "gptcache",
        top_k: int = 1,
        namespace: str = "",
    ):
        self._client = Elasticsearch("http://localhost:9200")
        self.top_k = top_k
        self.dimension = dimension
        self.collection_name = collection_name
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"
        self.create_collection(collection_name)

    def _create_collection(self, collection_name):
        if self._check_index_exists(collection_name):
            gptcache_log.info(
                "The %s already exists, and it will be used directly", collection_name
            )
        else:
            gptcache_log.info("Index does not exist")
            mappings = {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": self.dimension},
                }
            }
            self._client.indices.create(index=collection_name, mappings=mappings)

    def _check_index_exists(self, index_name):
        try:
            return self._client.exists(index=index_name)
        except NotFoundError:
            return False

    def mul_add(self, datas: List[VectorData]):
        for data in datas:
            id: int = data.id
            doc = {
                "_index": self.collection_name,
                "_id": id,
                "_source": {"vector": data.data.tolist()},
            }

    def search(self, data: np.ndarray, top_k: int = -1):
        search_body: dict = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                        "params": {"queryVector": data},
                    },
                }
            },
            "size": top_k,
            "sort": [{"_score": {"order": "desc"}}],
        }

        results = self._client.search(index=self.collection_name, body=search_body)
        return [
            (float(result["_score"]), int(result["_id"]))
            for result in results["hits"]["hits"]
        ]

    def rebuild(self, ids=None) -> bool:
        pass

    def delete(self, ids) -> None:
        for id in ids:
            self._client.delete(index=self.collection_name, id=id)