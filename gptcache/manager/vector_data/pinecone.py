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
            time.sleep(30)
        self.index = pinecone.Index(index_file_path)
        print("Information about index: ",pinecone.describe_index(index_file_path))
        self._top_k = top_k

    def mul_add(self, datas: List[VectorData], **kwargs):
        data_array, id_array, acc_id_array, pipeline_array = map(list, zip(*((data.data, data.id, data.account_id, data.pipeline) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        account_ids = np.array(acc_id_array)
        pipelines = np.array(pipeline_array)
        upsert_data = [(str(i_d), data.reshape(1,-1).tolist(), {'account_id': int(acc_id), 'pipeline': str(pipe)}) for (i_d,data,acc_id,pipe) in zip(ids,np_data,account_ids, pipelines)]
        self.index.upsert(upsert_data)
        
    def search(self, data: np.ndarray, top_k: int = -1, kwargs={}):
        print("<--------------------kwargs inside search of pinecone---------------------------->",kwargs)
        if self.index.describe_index_stats()['total_vector_count'] == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)
        response = self.index.query(vector = np_data.tolist(), top_k = top_k, include_values = False) #add additional filter
        dist, ids = [response['matches'][0]['score']], [int(response['matches'][0]['id'])]
        result = list(zip(dist, ids))
        # return list(zip(dist, ids))
        return result
    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        ids_to_remove = np.array(ids)
        self.index.delete(ids=ids_to_remove)  # add namespace

    def count(self):
        return self.index.describe_index_stats()['total_vector_count']
