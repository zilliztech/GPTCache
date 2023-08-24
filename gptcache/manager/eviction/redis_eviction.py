# pylint: disable=wrong-import-position
from abc import ABC
from typing import List

from gptcache.manager.eviction.distributed_cache import DistributedEviction
from gptcache.utils import import_redis


import_redis()
import redis
from redis_om import get_redis_connection


class RedisCacheEviction(DistributedEviction, ABC):
    """eviction: Distributed Cache Eviction Strategy using Redis.

    :param host: the host of redis
    :type host: str
    :param port: the port of redis
    :type port: int
    :param policy: eviction strategy policy of redis such as allkeys-lru, volatile-lru, allkeys-random, volatile-random, etc.
    refer https://redis.io/docs/reference/eviction/ for more information.
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]
    :param maxmemory: the maxmemory of redis
    :type maxmemory: str
    :param global_key_prefix: the global key prefix
    :type global_key_prefix: str
    :param ttl: the ttl of the cache data
    :type ttl: int
    :param maxmemory_samples: Number of keys to sample when evicting keys
    :type maxmemory_samples: int
    :param kwargs: the kwargs
    :type kwargs: Any
    """

    def __init__(self,
                 host="localhost",
                 port=6379,
                 maxmemory: str = None,
                 policy: str = None,
                 global_key_prefix="gptcache",
                 ttl: int = None,
                 maxmemory_samples: int = None,
                 **kwargs):
        self._redis = get_redis_connection(host=host, port=port, **kwargs)
        if maxmemory:
            self._redis.config_set("maxmemory", maxmemory)
        if maxmemory_samples:
            self._redis.config_set("maxmemory-samples", maxmemory_samples)
        if policy:
            self._redis.config_set("maxmemory-policy", policy)
            self._policy = policy.lower()

        self._global_key_prefix = global_key_prefix
        self._ttl = ttl

    def _create_key(self, key: str) -> str:
        return f"{self._global_key_prefix}:evict:{key}"

    def put(self, objs: List[str], expire=False):
        ttl = self._ttl if expire else None
        for key in objs:
            self._redis.set(self._create_key(key), "True", ex=ttl)

    def get(self, obj: str):

        try:
            value = self._redis.get(self._create_key(obj))
            # update key expire time when accessed
            if self._ttl:
                self._redis.expire(self._create_key(obj), self._ttl)
            return value
        except redis.RedisError:
            print(f"Error getting key {obj} from cache")
            return None

    @property
    def policy(self) -> str:
        return self._policy
