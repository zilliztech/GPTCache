from gptcache.utils import import_pymilvus
import_pymilvus()

from uuid import uuid4
import numpy as np
from .base import VectorBase, ClearStrategy
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
    MilvusException
)


class Milvus(VectorBase):

    def __init__(self, **kwargs):
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 19530)
        user = kwargs.get("user", "")
        password = kwargs.get("password", "")
        use_security = kwargs.get("is_https", False)

        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
        }

        try:
            i = [
                connections.get_connection_addr(x[0])
                for x in connections.list_connections()
            ].index({"host": host, "port": port})
            self.alias = connections.list_connections()[i][0]
        except ValueError:
            # Connect to the Milvus instance using the passed in Environment variables
            self.alias = uuid4().hex
            connections.connect(
                alias=self.alias,
                host=host,
                port=port,
                user=user,  # type: ignore
                password=password,  # type: ignore
                secure=use_security,
            )
        collection_name = kwargs.get("collection_name", "gptcache")
        create_new = kwargs.get("create_new", False)
        dim = kwargs.get("dim", 0)
        if dim <= 0:
            raise ValueError(f"invalid dim param `{dim}` in the Milvus vector store")
        self.dim = dim
        self.top_k = kwargs.get("top_k", 1)
        self.index_params = kwargs.get("index_params", None)
        self._create_collection(collection_name, create_new)

        self.search_params = (
                kwargs.get("search_params", None) or self.default_search_params[self.index_params["index_type"]]
        )

    def _create_collection(self, collection_name, create_new):
        if utility.has_collection(collection_name, using=self.alias) and create_new:
            utility.drop_collection(collection_name, using=self.alias)

        if utility.has_collection(collection_name, using=self.alias) is False:
            schema = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(schema)
            self.col = Collection(
                collection_name,
                schema=schema,
                consistency_level="Strong",
                using=self.alias,
            )
        else:
            self.col = Collection(
                collection_name, consistency_level="Strong", using=self.alias
            )

        if len(self.col.indexes) == 0:
            if self.index_params is not None:
                self.col.create_index("embedding", index_params=self.index_params)
            else:
                try:
                    print("Attempting creation of Milvus default index")
                    i_p = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }

                    self.col.create_index("embedding", index_params=i_p)
                    self.index_params = i_p
                    print("Creation of Milvus default index successful")
                except MilvusException as e:
                    print("milvus e", e)
                    print("Attempting creation of default index")
                    i_p = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
                    self.col.create_index("embedding", index_params=i_p)
                    self.index_params = i_p
        else:
            self.index_params = self.col.indexes[0].to_dict()["index_param"]

        self.col.load()

    def add(self, key: str, data: np.ndarray):
        entities = [
            [key],
            data.reshape(1, self.dim)
        ]
        self.col.insert(entities)

    def search(self, data: np.ndarray):
        search_result = self.col.search(
            data=data.reshape(1, -1).tolist(),
            anns_field="embedding",
            param=self.search_params,
            limit=self.top_k,
            output_fields=[
                "pk"
            ]
        )
        pks = {}
        for hit in search_result[0]:
            pks[hit.entity.get("pk")] = hit.score

        query_pks = ",".join(['"' + x + '"' for x in pks])
        query_result = self.col.query(expr=f"pk in [{query_pks}]", output_fields=["pk", "embedding"])

        search_tuples = []
        for query_row in query_result:
            search_tuples.append((pks[query_row["pk"]], np.array(query_row["embedding"])))
        return search_tuples

    def clear_strategy(self):
        return ClearStrategy.DELETE

    def delete(self, ids):
        pks = ",".join(['"' + x + '"' for x in ids])
        self.col.delete(f"pk in [{pks}]")

    def close(self):
        self.col.flush(_async=True)
