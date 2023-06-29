from typing import List
from uuid import uuid4
import numpy as np

from gptcache.utils import import_pymilvus
from gptcache.utils.log import gptcache_log
from gptcache.manager.vector_data.base import VectorBase, VectorData


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
    """vector store: Milvus

    :param host: the host for Milvus vector database, defaults to 'localhost'.
    :type host: str
    :param port: the port for Milvus vector database, defaults to '19530'.
    :type port: str
    :param user: the user for Zilliz Cloud, defaults to "".
    :type user: str
    :param password: the password for Zilliz Cloud, defaults to "".
    :type password: str
    :param secure: whether it is https with Zilliz Cloud, defaults to False.
    :type secures: bool
    :param collection_name: the name of the collection for Milvus vector database, defaults to 'gptcache'.
    :type collection_name: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param index_params: the index parameters for Milvus, defaults to the HNSW index: {'metric_type': 'L2', 'index_type': 'HNSW', 'params': {'M':
                         8, 'efConstruction': 64}}.
    :type index_params: dict
    :param local_mode: if true, will start a local milvus server.
    :type local_mode: bool
    :param local_data: required when local_mode is True.
    :type local_data: str
    """

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
        local_mode: bool = False,
        local_data: str = "./milvus_data"
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Milvus vector store."
            )
        self._local_mode = local_mode
        self._local_data = local_data
        self.dimension = dimension
        self.top_k = top_k
        self.index_params = index_params
        if self._local_mode:
            self._create_local(port, local_data)
        self._connect(host, port, user, password, secure)
        self._create_collection(collection_name)
        self.search_params = (
            search_params or self.SEARCH_PARAM[self.index_params["index_type"]]
        )

    def _create_local(self, port, local_data):
        from gptcache.utils import import_milvus_lite  # pylint: disable=import-outside-toplevel
        import_milvus_lite()
        from milvus import MilvusServer  # pylint: disable=import-outside-toplevel
        self._server = MilvusServer()
        self._server.set_base_dir(local_data)
        self._server.listen_port = int(port)
        self._server.start()

    def _connect(self, host, port, user, password, secure):
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
                timeout=10
            )

    def _create_collection(self, collection_name):
        if not utility.has_collection(collection_name, using=self.alias):
            schema = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
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
            gptcache_log.warning("The %s collection already exists, and it will be used directly.", collection_name)
            self.col = Collection(
                collection_name, consistency_level="Strong", using=self.alias
            )

        if len(self.col.indexes) == 0:
            try:
                gptcache_log.info("Attempting creation of Milvus index.")
                self.col.create_index("embedding", index_params=self.index_params)
                gptcache_log.info("Creation of Milvus index successful.")
            except MilvusException as e:
                gptcache_log.warning("Error with building index: %s, and attempting creation of default index.", e)
                i_p = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
                self.col.create_index("embedding", index_params=i_p)
                self.index_params = i_p
        else:
            self.index_params = self.col.indexes[0].to_dict()["index_param"]

        self.col.load()

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        entities = [id_array, np_data]
        self.col.insert(entities)

    def search(self, data: np.ndarray, top_k: int = -1):
        if top_k == -1:
            top_k = self.top_k
        search_result = self.col.search(
            data=data.reshape(1, -1).tolist(),
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
        )
        return list(zip(search_result[0].distances, search_result[0].ids))

    def delete(self, ids):
        del_ids = ",".join([str(x) for x in ids])
        self.col.delete(f"id in [{del_ids}]")

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        self.col.compact()

    def flush(self):
        self.col.flush(_async=True)

    def close(self):
        self.flush()
        if self._local_mode:
            self._server.stop()

    def get_embeddings(self, data_id: int):
        vec_emb = self.col.query(
            expr=f"id == {data_id}",
            output_fields=["embedding"],
        )
        if len(vec_emb) < 1:
            return None
        vec_emb = np.asarray(vec_emb[0]["embedding"], dtype="float32")
        return vec_emb

    def update_embeddings(self, data_id: int, emb: np.ndarray):
        self.col.delete(f"id in [{data_id}]")
        data = [
            [data_id],
            np.expand_dims(emb, axis=0),
        ]
        self.col.insert(data)
