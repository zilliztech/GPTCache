from gptcache import Cache
from gptcache.embedding import Onnx

from gptcache.manager.eviction import EvictionBase

from gptcache.manager import get_data_manager, CacheBase, VectorBase, manager_factory


def get_data_manager_example():
    """
    This example shows how to create a data manager with a mongo as a scalar storage, faiss vector base,
    and redis eviction base.
    This type of configuration can be used to scale GPTCache horizontally.
    Where keys will be maintained in redis key-value store instead of in-memory.
    The eviction of the keys will be handled based on the eviction policy of redis.
    """
    onnx = Onnx()
    data_manager = get_data_manager(cache_base=CacheBase("mongo", url="mongodb://localhost:27017/"),
                                    vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                    eviction_base=EvictionBase("redis",
                                                               maxmemory="100mb",
                                                               policy="allkeys-lru",
                                                               ttl=100))

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "What is github?"
    answer = "Online platform for version control and code collaboration."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])


def get_manager_example_redis_only():
    """
    Note: Since, `RedisScalarStorage` can be configured to internally handle the ttl of the keys and their eviction.
    In this scenario, `no_op_eviction` is used as the eviction base. It will not add any keys or update their ttls.

    This example shows how to create a data manager with a redis as a scalar storage, as well as eviction base.
    This type of configuration can be used to scale GPTCache horizontally.
    Where keys will be maintained in redis key-value store instead of in-memory.
    The eviction of the keys will be handled based on the eviction policy of redis.

    """
    onnx = Onnx()
    data_manager = get_data_manager(cache_base=CacheBase("redis", maxmemory="100mb", policy="allkeys-lru", ttl=100),
                                    vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                    eviction_base=EvictionBase("no_op_eviction"))

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "What is github?"
    answer = "Online platform for version control and code collaboration."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])


def manager_factory_example():
    onnx = Onnx()
    data_manager = manager_factory("redis,faiss",
                                   eviction_manager="redis",
                                   scalar_params={"url": "redis://localhost:6379"},
                                   vector_params={"dimension": onnx.dimension},
                                   eviction_params={"maxmemory": "100mb",
                                                    "policy": "allkeys-lru",
                                                    "ttl": 1}
                                   )

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "What is github?"
    answer = "Online platform for version control and code collaboration."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])
