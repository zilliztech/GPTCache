import os
from pathlib import Path
from typing import Union, Callable

from gptcache.manager import CacheBase, VectorBase, ObjectBase
from gptcache.manager.data_manager import SSDataManager, MapDataManager
from gptcache.manager.eviction import EvictionBase
from gptcache.utils.log import gptcache_log


def manager_factory(manager="map",
                    data_dir="./",
                    max_size=1000,
                    eviction_manager: str = "memory",
                    get_data_container: Callable = None,
                    scalar_params=None,
                    vector_params=None,
                    object_params=None,
                    eviction_params=None
                    ):
    """Factory of DataManager.
       By using this factory method, you only need to specify the root directory of the data,
       and it can automatically manage all the local files.

    :param manager: Type of DataManager. Supports: Map, or {scalar_name},{vector_name}
                    or {scalar_name},{vector_name},{object_name}
    :type manager: str
    :param data_dir: Root path for data storage.
    :type data_dir: str
    :param max_size: the max size for the LRU cache in MapDataManager, defaults to 1000.
    :type max_size: int
    :param eviction_manager: The eviction manager, defaults to "memory".
                             It supports "memory" and "redis" and 'no_op_eviction'.
    :type eviction_manager:  str
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable

    :param scalar_params: Params of scalar storage.
    :type scalar_params:  dict

    :param vector_params: Params of vector storage.
    :type vector_params:  dict

    :param object_params: Params of object storage.
    :type object_params:  dict

    :param eviction_params: Params of eviction.
    :type eviction_params:  dict
    :return: SSDataManager or MapDataManager.

    Example:
        .. code-block:: python

            from gptcache.manager import manager_factory

            data_manager = manager_factory("sqlite,faiss", data_dir="./workspace", vector_params={"dimension": 128})

            # or using manager factory enabled with redis cache instead of in-memory cache
            from gptcache.manager import manager_factory
            data_manager = manager_factory("redis,faiss",
                                           eviction_manager="redis",
                                           scalar_params={"maxmemory": "2mb",
                                                          "policy": "allkeys-lru"
                                                          },
                                           vector_params={"dimension": 128},
                                           eviction_params=dict(url="redis://localhost:6379")
                                           )
    """

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manager = manager.lower()

    if manager == "map":
        return MapDataManager(os.path.join(data_dir, "data_map.txt"), max_size, get_data_container)

    db_infos = manager.split(",")
    if len(db_infos) not in [2, 3]:
        raise RuntimeError(
            "Error manager format: %s, the correct is \"{scalar},{vector},{object}\", object is optional" % manager)

    if len(db_infos) == 2:
        db_infos.append("")
    scalar, vector, obj = db_infos

    if scalar_params is None:
        scalar_params = {}
    if scalar == "sqlite":
        scalar_params["sql_url"] = "sqlite:///" + os.path.join(data_dir, "sqlite.db")
    s = CacheBase(name=scalar, **scalar_params)

    if vector_params is None:
        vector_params = {}
    local_vector_type = ["faiss", "hnswlib", "docarray"]
    if vector in local_vector_type:
        vector_params["index_path"] = os.path.join(data_dir, f"{vector}.index")
    elif vector == "milvus" and vector_params.get("local_mode", False) is True:
        vector_params["local_data"] = os.path.join(data_dir, "milvus_data")
    v = VectorBase(name=vector, **vector_params)

    if object_params is None:
        object_params = {}
    if obj == "local":
        object_params["path"] = os.path.join(data_dir, "local_obj")
    o = ObjectBase(name=obj, **object_params) if obj else None

    if eviction_params is None:
        eviction_params = {}

    if  scalar == "redis" and eviction_manager == "redis":
        # if cache manager and eviction manager are both redis, we use no op redis to avoid redundant operations
        eviction_manager = "no_op_eviction"
        gptcache_log.info("Since Scalar Storage and Eviction manager are both redis, "
                          "no_op_eviction will be used to avoid redundant operations.")
        s.init_eviction_params(
            maxmemory=eviction_params.get("maxmemory", scalar_params.get("maxmemory")),
            policy=eviction_params.get("policy", scalar_params.get("policy")),
            ttl=eviction_params.get("ttl", scalar_params.get("ttl")),
            maxmemory_samples=eviction_params.get("maxmemory_samples", scalar_params.get("maxmemory_samples")),
        )

    e = EvictionBase(
        name=eviction_manager,
        **eviction_params
    )
    return get_data_manager(s, v, o, e)


def get_data_manager(
        cache_base: Union[CacheBase, str] = None,
        vector_base: Union[VectorBase, str] = None,
        object_base: Union[ObjectBase, str] = None,
        eviction_base: Union[EvictionBase, str] = None,
        max_size: int = 1000,
        clean_size=None,
        eviction: str = "LRU",
        data_path: str = "data_map.txt",
        get_data_container: Callable = None,
):
    """Generate `SSDataManager` (with `cache_base`, `vector_base`, `max_size`, `clean_size` and `eviction` params),
       or `MAPDataManager` (with `data_path`, `max_size` and `get_data_container` params) to manager the data.

    :param cache_base: a CacheBase object, or the name of the cache storage, it is support 'sqlite', 'duckdb', 'postgresql',
                       'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type cache_base: :class:`CacheBase` or str
    :param vector_base: a VectorBase object, or the name of the vector storage, it is support 'milvus', 'faiss' and
                        'chromadb' now.
    :type vector_base: :class:`VectorBase` or str
    :param object_base: a object storage, supports local path and s3.
    :type object_base: :class:`ObjectBase` or str
    :param max_size: the max size for the LRU cache in MapDataManager, defaults to 1000.
    :type max_size: int
    :param eviction_base: a EvictionBase object, or the name of the eviction, it supports:
     - 'memory'
     - 'redis'
     - 'no_op_eviction'.
    :type eviction_base: :class:`EvictionBase` or str
    :param clean_size: the clean size for the LRU cache in MapDataManager, defaults to None.
    :type clean_size: int
    :param eviction: the eviction policy for the LRU cache in MapDataManager, defaults to 'LRU'.
    :type eviction: str
    :param data_path: the path to save the map data, defaults to 'data_map.txt'.
    :type data_path:  str
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable


    :return: SSDataManager or MapDataManager.

    Example:
        .. code-block:: python

            from gptcache.manager import get_data_manager, CacheBase, VectorBase

            data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))

            # or using manager factory enabled with redis cache instead of in-memory cache
            # example 1: using redis eviction base with sqlite cache base
            data_manager = get_data_manager(cache_base=CacheBase("sqlite"),
                                        vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                        eviction_base=EvictionBase("redis",  maxmemory="0", policy="noeviction", ttl=1))

           # example 2: using redis eviction base with redis cache base
           # here no_op_eviction is used since `redis` cache base already handles evictions internally
            data_manager = get_data_manager(cache_base=CacheBase("redis",  maxmemory="0", policy="noeviction", ttl=1),
                                        vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                        eviction_base=EvictionBase("no_op_eviction"))
    """
    if not cache_base and not vector_base:
        return MapDataManager(data_path, max_size, get_data_container)

    if isinstance(cache_base, str):
        cache_base = CacheBase(name=cache_base)
    if isinstance(vector_base, str):
        vector_base = VectorBase(name=vector_base)
    if isinstance(object_base, str):
        object_base = ObjectBase(name=object_base)
    if isinstance(eviction_base, str):
        eviction_base = EvictionBase(name=eviction_base)
    assert cache_base and vector_base
    return SSDataManager(cache_base, vector_base, object_base, eviction_base, max_size, clean_size, eviction)
