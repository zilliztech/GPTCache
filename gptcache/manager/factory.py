from typing import Union, Callable
from gptcache.manager.data_manager import SSDataManager, MapDataManager
from gptcache.manager import CacheBase, VectorBase


def get_data_manager(
    cache_base: Union[CacheBase, str] = None,
    vector_base: Union[VectorBase, str] = None,
    max_size: int = 1000,
    clean_size: int = None,
    eviction: str = "LRU",
    data_path: str = "data_map.txt",
    get_data_container: Callable = None,
):
    """Generate `SSDataManager` (with `cache_base`, `vector_base`, `max_size`, `clean_size` and `eviction` params),
       or `MAPDataManager` (with `data_path`, `max_size` and `get_data_container` params) to manager the data.

    :param cache_base: a CacheBase object, or the name of the cache storage, it is support 'sqlite', 'postgresql',
                       'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type cache_base: :class:`CacheBase` or str
    :param vector_base: a VectorBase object, or the name of the vector storage, it is support 'milvus', 'faiss' and
                        'chromadb' now.
    :type vector_base: :class:`VectorBase` or str
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param clean_size: the size to clean up, defaults to `max_size * 0.2`.
    :type clean_size: int
    :param eviction: the eviction policy, it is support "LRU" and "FIFO" now, and defaults to "LRU".
    :type eviction:  str
    :param data_path: the path to save the map data, defaults to 'data_map.txt'.
    :type data_path:  str
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable


    :return: SSDataManager or MapDataManager.

    Example:
        .. code-block:: python

            from gptcache.manager import get_data_manager, CacheBase, VectorBase

            data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
    """
    if not cache_base and not vector_base:
        return MapDataManager(data_path, max_size, get_data_container)

    if isinstance(cache_base, str):
        cache_base = CacheBase(name=cache_base)
    if isinstance(vector_base, str):
        vector_base = VectorBase(name=cache_base)
    assert cache_base and vector_base
    return SSDataManager(cache_base, vector_base, max_size, clean_size, eviction)
