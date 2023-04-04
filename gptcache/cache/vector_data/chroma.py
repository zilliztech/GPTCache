from gptcache.utils import import_chromadb
import_chromadb()

import chromadb
from .base import VectorBase, ClearStrategy


class Chromadb(VectorBase):
    def __init__(self, **kwargs):
        client_settings = kwargs.get("client_settings", None)
        persist_directory = kwargs.get("persist_directory", None)
        collection_name = kwargs.get('collection_name', 'gptcache')
        self.top_k = kwargs.get("top_k", 1)

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
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )

    def add(self, key, data):
      self._collection.add(
          embeddings=[data], ids=[key]
      )

    def search(self, data):
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[data], n_results=self.top_k,
            include=['distances', 'embeddings']
           )
        return list(zip(results['distances'][0], results['embeddings'][0]))

    def clear_strategy(self):
        return ClearStrategy.DELETE    

    def delete(self, ids):
        self._collection.delete(ids)

    def close(self):
        return True
