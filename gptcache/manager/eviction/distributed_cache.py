from abc import ABC
from typing import Any, List, Callable

import redis
from redis_om import get_redis_connection

from gptcache.manager.eviction.base import EvictionBase
from gptcache.manager.scalar_data.redis_storage import RedisCacheStorage


class DistributedCacheEviction(EvictionBase, ABC):
    """eviction: Distributed Cache
    :param host: the host of redis
    :type host: str
    :param port: the port of redis
    :type port: int
    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]
    :param maxmemory: the maxmemory of redis
    :type maxmemory: str

    """

    def __init__(self,
                 host='localhost',
                 port=6379,
                 maxmemory: str = '100mb',
                 policy: str = 'allkeys-lru',
                 redis_cache_storage: RedisCacheStorage = None,
                 **kwargs):
        self._redis = get_redis_connection(host=host, port=port, **kwargs)
        self._redis.config_set('maxmemory', maxmemory)
        self._redis.config_set('maxmemory-policy', policy)
        self._policy = policy.lower()
        self._s = redis_cache_storage

    def put(self, objs: List[Any], ttl: int = None):
        if not self._s:
            for obj in objs:
                self._redis.set(obj, "True")

    def get(self, key: str):
        if self._s:
            return self._s.get_data_by_id(key)

        try:
            value = self._redis.get(key)
            return value
        except redis.RedisError:
            print(f"Error getting key {key} from cache")
            return None

    @property
    def policy(self) -> str:
        return self._policy
