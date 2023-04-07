from uuid import uuid4
import numpy as np

from gptcache.utils import import_pymilvus
from gptcache.manager.vector_data.base import VectorBase, ClearStrategy

import_pymilvus()

from pymilvus import (  # pylint: disable=C0413
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
    MilvusException,
)


class Milvus(VectorBase):
    """vector store: Milvus"""

    SEARCH_PARAM = {
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

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        secure: bool = False,
        collection_name: str = "gptcache",
        dimension: int = 0,
        top_k: int = 1,
        index_params: dict = None,
        search_params: dict = None,
    ):
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
                secure=secure,
            )
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Milvus vector store."
            )
        self.dimension = dimension
        self.top_k = top_k
        self.index_params = index_params
        self._create_collection(collection_name)
        self.search_params = (
            search_params or self.SEARCH_PARAM[self.index_params["index_type"]]
        )

    def _create_collection(self, collection_name):
        if not utility.has_collection(collection_name, using=self.alias):
            schema = [
                FieldSchema(
                    name="pk",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=65535,
                ),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                ),
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
            try:
                print("Attempting creation of Milvus index")
                self.col.create_index("embedding", index_params=self.index_params)
                print("Creation of Milvus index successful")
            except MilvusException as e:
                print("Error with building index: ", e)
                print("Attempting creation of default index")
                i_p = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
                self.col.create_index("embedding", index_params=i_p)
                self.index_params = i_p
        else:
            self.index_params = self.col.indexes[0].to_dict()["index_param"]

        self.col.load()

    def add(self, key: str, data: np.ndarray):
        entities = [[key], data.reshape(1, self.dimension)]
        self.col.insert(entities)

    def search(self, data: np.ndarray):
        search_result = self.col.search(
            data=data.reshape(1, -1).tolist(),
            anns_field="embedding",
            param=self.search_params,
            limit=self.top_k,
            output_fields=["pk"],
        )
        pks = {}
        for hit in search_result[0]:
            pks[hit.entity.get("pk")] = hit.score

        query_pks = ",".join(["'" + x + "'" for x in pks])
        query_result = self.col.query(
            expr=f"pk in [{query_pks}]", output_fields=["pk", "embedding"]
        )

        search_tuples = []
        for query_row in query_result:
            search_tuples.append(
                (pks[query_row["pk"]], np.array(query_row["embedding"]))
            )
        return search_tuples

    def clear_strategy(self):
        return ClearStrategy.DELETE

    def delete(self, ids):
        pks = ",".join(["'" + x + "'" for x in ids])
        self.col.delete(f"pk in [{pks}]")

    def close(self):
        self.col.flush(_async=True)
