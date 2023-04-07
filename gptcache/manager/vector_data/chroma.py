from gptcache.manager.vector_data.base import VectorBase, ClearStrategy
from gptcache.utils import import_chromadb

import_chromadb()

import chromadb  # pylint: disable=C0413


class Chromadb(VectorBase):
    """vector store: Chromadb"""

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

    def add(self, key, data):
        self._collection.add(embeddings=[data], ids=[key])

    def search(self, data):
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[data],
            n_results=self.top_k,
            include=["distances", "embeddings"],
        )
        return list(zip(results["distances"][0], results["embeddings"][0]))

    def clear_strategy(self):
        return ClearStrategy.DELETE

    def delete(self, ids):
        self._collection.delete(ids)

    def close(self):
        return True
