import numpy as np

from gptcache.embedding import Onnx
from gptcache.manager import VectorBase
from gptcache.manager.vector_data.base import VectorData


def test_redis_vector_store():
    encoder = Onnx()
    dim = encoder.dimension
    vector_base = VectorBase("redis", dimension=dim)
    vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(10)])

    search_res = vector_base.search(np.random.rand(dim))
    print(search_res)
    assert len(search_res) == 1

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 10

    vector_base.delete([i for i in range(5)])

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 5
