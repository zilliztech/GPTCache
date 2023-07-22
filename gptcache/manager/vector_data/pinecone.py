from importlib.metadata import metadata
import os
from typing import List
from xml.etree.ElementInclude import include
import numpy as np
from gptcache.manager.vector_data.base import VectorBase, VectorData
import pinecone
import time

class Pinecone(VectorBase):
    """vector store: Pinecone

    :param index_path: the path to Pinecone index, defaults to 'caching'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    """

    def __init__(self, index_file_path, dimension, top_k, metric):
        self._index_file_path = index_file_path
        self._dimension = dimension
        assert metric=='euclidean'
        self.indexes = pinecone.list_indexes()
        if index_file_path not in self.indexes:
            pinecone.create_index(index_file_path, dimension=dimension, metric=metric)
            time.sleep(50)
        self.index = pinecone.Index(index_file_path)
        self._top_k = top_k

    def mul_add(self, datas: List[VectorData], **kwargs):
        metadata = kwargs.get('kwargs').get('kwargs').pop('metadata',{}) 
        assert metadata!={}, "Please provide the metadata for the following request to process!!"
        data_array, id_array= map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        upsert_data = [(str(i_d), data.reshape(1,-1).tolist(), {"account_id": int(metadata['account_id']), "pipeline": str(metadata['pipeline'])}) for (i_d,data) in zip(ids,np_data)]
        self.index.upsert(upsert_data)
        
    def search(self, data: np.ndarray, top_k: int = -1, **kwargs):
        if self.index.describe_index_stats()['total_vector_count'] == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        metadata = kwargs.get("metadata",{})
        assert metadata!={}, "Please provide metadata for the search query!!"
        np_data = np.array(data).astype("float32").reshape(1, -1)
        response = self.index.query(vector = np_data.tolist(), top_k = top_k, include_values = False, filter={"account_id": int(metadata["account_id"]), "pipeline": str(metadata["pipeline"])}) #add additional filter
        if len(response['matches'])!=0:
            dist, ids = [response['matches'][0]['score']], [int(response['matches'][0]['id'])]
            return list(zip(dist, ids))
        else: 
            return None
    
    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        ids_to_remove = np.array(ids)
        self.index.delete(ids=ids_to_remove)  # add namespace

    def count(self):
        return self.index.describe_index_stats()['total_vector_count']
