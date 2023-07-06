from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_weaviate
from gptcache.utils.log import gptcache_log

from weaviate import Client, EmbeddedOptions, Config

import_weaviate()

class Weaviate(VectorBase):
    """Weaviate Vector store"""
    def __init__(self, 
                 url: str | None = None,
                 auth_client_secret: None = None,
                 timeout_config = (10, 60),
                 proxies: dict | str | None = None,
                 trust_env: bool = False,
                 additional_headers: dict | None = None,
                 startup_period: int | None = 5,
                 embedded_options: None = None,
                 additional_config: None = None,
                 top_k: int = 1,
                 distance: str = "cosine",
                 collection_name: str = "Gptcache",
                 ):
        self.class_name = collection_name
        self.top_k = top_k
        self.distance = distance
        if embedded_options:
            self.client = Client(embedded_options = EmbeddedOptions(),
                                 startup_period = startup_period,
                                 timeout_config = timeout_config,
                                 additional_config=additional_config)
        else:
            self.client = Client(url, 
                                 auth_client_secret,
                                 timeout_config,
                                 proxies,
                                 trust_env,
                                 additional_headers,
                                 startup_period,
                                 embedded_options,
                                 additional_config,
                                 )        
    
    def _create_collection(self, class_name: str):
        if not class_name:
            class_name = self.class_name
        if self.client.schema.exists(class_name):
            gptcache_log.info(
                "The %s already exists, and it will be used directly", class_name
            )
        else:   
            gptcache_class_schema = {
                  "class": class_name,
                  "description": "caching LLM responses",
                  "properties": [
                      {
                          "name": "id_",
                          "dataType": ["int"],
                      }
                      ],
                    'vectorIndexConfig': 
                    {
                        "distance": self.distance
                    }                    
                  }
        self.client.schema.create_class(gptcache_class_schema)

    def mul_add(self, datas: List[VectorData]):
        with self.client.batch(
            batch_size=len(datas)
            ) as batch:
            # Batch import 
            for data in datas:           
                properties = {
                        "id_": data.id,
                    }            
                self.client.batch.add_data_object(
                    properties,
                    self.class_name,
                    vector = data.data.tolist()
                    )
    
    def search(self, data: np.ndarray, top_k: int = -1):
        if not self.client.schema.exists(self.class_name):
            self._create_collection(self.class_name)
        if top_k==-1:
            top_k = self.top_k
        result = self.client.query.get(class_name = self.class_name, properties = ['id_']).\
                                   with_near_vector(content={"vector": data.tolist()}).\
                                   with_additional(['distance']).\
                                   with_limit(top_k).do()
        return list(map(lambda x: (x['_additional']['distance'], x['id_']), result['data']['Get'][self.class_name]))

    def get_uuids(self, ids: List[str]):
        uuid_list = []
        for id_ in ids:
            res = self.client.query.get(class_name=self.class_name, properties=['id_']).\
                                    with_where({"path": ["id_"], "operator":"Equal", "valueNumber":id_}).\
                                    with_additional(["id"]).do()
            uuid_list.append(res['data']['Get'][self.class_name][0]['_additional']['id'])
        return uuid_list

    def delete(self, ids: List[str]):
        uuids = self.get_uuids(ids)
        for uuid_ in uuids:
            self.client.data_object.delete(class_name='example', uuid=uuid_)

    def rebuild(self, ids=None) :
        return 
    
    def flush(self):
        return True
    
    def close(self):
        pass



        

    
