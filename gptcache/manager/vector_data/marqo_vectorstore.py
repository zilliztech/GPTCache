from typing import List, Optional

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_marqo
from gptcache.utils.log import gptcache_log

import_marqo()
from marqo import Client

class MarqoVectorStore(VectorBase):
    """
    Marqo Vector Store - https://github.com/marqo-ai/marqo
    """

    def __init__(self, 
                 marqo_url, 
                 index_name: Optional[str] = "gptcache", 
                 dimension: Optional[int] = 0, 
                 top_k: int = 1,
                 ):
        self.top_k = top_k
        self._client = Client(
            url=marqo_url
        )
        self._index_name = index_name
        self._dimension = dimension
        self._index_settings_dict = {
                    'index_defaults': {
                        'model': 'no_model',
                        'model_properties': {
                            'dimensions': self._dimension 
                        },
                        'ann_parameters':{
                            'space_type': 'cosinesimil'
                        }
                    }
                }

        self._client.create_index(
            index_name=self._index_name,
            settings_dict=self._index_settings_dict)
        

    def mul_add(self, datas: List[VectorData]):
        
        vecs = []
        for d in datas:
            vecs.append(
                {
                    '_id': str(d.id),
                    'gptcachevec':{
                        'vector': d.data.tolist()
                    }
                }
            )
        self._client.index(self._index_name).add_documents(
            documents=vecs,
            mappings={
                'gptcachevec': 
                    {
                    'type': 'custom_vector'
                    }
            },
            tensor_fields=['gptcachevec'],
            auto_refresh=True
        )
    
    def search(self, data: np.ndarray, top_k: int = -1):
        if self._client.index(self._index_name).get_stats()['numberOfDocuments']==0:
            return []
        if top_k == -1:
            top_k = self.top_k
        
        search_results = self._client.index(self._index_name).search(
                context={
                    'tensor':[{'vector': data.tolist(), 'weight' : 1}]
                },
                limit=top_k,
            )['hits']
        return [(r['_id'], r['_score']) for r in search_results]
        
    def get_embeddings(self, data_id: int | str) -> np.ndarray | None:
        
        doc = self._client.index(self._index_name).get_document(
            document_id=str(data_id),
            expose_facets=True
        )
        embedding = doc['_tensor_facets'][0]['_embedding']
        return embedding
    
    def delete(self, ids) -> bool:
        self._client.index(self._index_name).delete_documents(
                ids=[str(_id) for _id in ids]
            )
        
    def rebuild(self, ids=None) -> bool:
        return
    
    def flush(self):
        pass

    def close(self):
        return self.flush()