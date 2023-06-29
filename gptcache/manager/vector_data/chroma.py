from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_chromadb, import_torch

import_torch()
import_chromadb()

import chromadb  # pylint: disable=C0413


class Chromadb(VectorBase):
    """vector store: Chromadb

    :param client_settings: the setting for Chromadb.
    :type client_settings: Settings
    :param persist_directory: the directory to persist, defaults to .chromadb/ in the current directory.
    :type persist_directory: str
    :param collection_name: the name of the collection in Chromadb, defaults to 'gptcache'.
    :type collection_name: str
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int

    """

    def __init__(
        self,
        client_settings=None,
        persist_directory=None,
        collection_name: str = "gptcache",
        top_k: int = 1,
    ):
        self.top_k = top_k

        if client_settings:
            self._client_settings = client_settings
        else:
            self._client_settings = chromadb.config.Settings()
            if persist_directory is not None:
                self._client_settings = chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
                )
        self._client = chromadb.Client(self._client_settings)
        self._persist_directory = persist_directory
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        self._collection.add(embeddings=data_array, ids=id_array)

    def search(self, data, top_k: int = -1):
        if self._collection.count() == 0:
            return []
        if top_k == -1:
            top_k = self.top_k
        results = self._collection.query(
            query_embeddings=[data.tolist()],
            n_results=top_k,
            include=["distances"],
        )
        return list(zip(results["distances"][0], [int(x) for x in results["ids"][0]]))

    def delete(self, ids):
        self._collection.delete([str(x) for x in ids])

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        return True

    def get_embeddings(self, data_id: str):
        vec_emb = self._collection.get(
                str(data_id),
                include=["embeddings"],
            )["embeddings"]
        if vec_emb is None or len(vec_emb) < 1:
            return None
        vec_emb = np.asarray(vec_emb[0], dtype="float32")
        return vec_emb

    def update_embeddings(self, data_id: str, emb: np.ndarray):
        self._collection.update(
            ids=str(data_id),
            embeddings=emb.tolist(),
        )
