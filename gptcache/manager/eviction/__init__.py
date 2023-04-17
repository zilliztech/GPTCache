__all__ = ["EvictionBase"]

from gptcache.utils.lazy_import import LazyImport

eviction_manager = LazyImport(
    "eviction_manager", globals(), "gptcache.manager.eviction.manager"
)


def EvictionBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration.

    :param name: the name of the eviction, like: memory
    :type name: str

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]

    Example:
        .. code-block:: python

            from gptcache.manager import EvictionBase

            cache_base = EvictionBase('memory', policy='lru', maxsize=10, clean_size=2, on_evict=lambda x: print(x))
    """
    return eviction_manager.EvictionBase.get(name, **kwargs)
