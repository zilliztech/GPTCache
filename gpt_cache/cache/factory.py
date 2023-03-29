from .data_manager import DataManager, SIDataManager, SSDataManager
from .scalar_data.sqllite3 import SQLite
from .vector_data.faiss import Faiss
from .vector_data.milvus import Milvus


def get_data_manager(data_manager_name: str, **kwargs) -> DataManager:
    if data_manager_name == "map":
        from .data_manager import MapDataManager

        return MapDataManager(kwargs.get("data_path", "data_map.txt"),
                              kwargs.get("max_size", 100))
    elif data_manager_name == "scalar_vector":
        scalar_store = kwargs.get("scalar_store", None)
        vector_store = kwargs.get("vector_store", None)
        max_size = kwargs.get("max_size", 1000)
        clean_size = kwargs.get("clean_size", int(max_size * 0.2))
        if scalar_store is None or vector_store is None:
            raise ValueError(f"Missing scalar_store or vector_store parameter for scalar_vector")
        return SSDataManager(max_size, clean_size, scalar_store, vector_store)
    elif data_manager_name == "scalar_vector_index":
        scalar_store = kwargs.get("scalar_store", None)
        vector_index = kwargs.get("vector_index", None)
        max_size = kwargs.get("max_size", 1000)
        clean_size = kwargs.get("clean_size", int(max_size * 0.2))
        if scalar_store is None or vector_index is None:
            raise ValueError(f"Missing scalar_store or vector_index parameter for scalar_vector_index")
        return SIDataManager(max_size, clean_size, scalar_store, vector_index)
    # elif data_manager_name == "sqlite_faiss":
    #     from .data_manager import SFDataManager
    #
    #     dimension = kwargs.get("dimension", 0)
    #     if dimension <= 0:
    #         raise ValueError(f"the sqlite_faiss data manager should set the 'dimension' parameter greater than zero, "
    #                          f"current: {dimension}")
    #     top_k = kwargs.get("top_k", 1)
    #     sqlite_path = kwargs.get("sqlite_path", "sqlite.db")
    #     index_path = kwargs.get("index_path", "faiss.index")
    #     max_size = kwargs.get("max_size", 1000)
    #     clean_size = kwargs.get("clean_size", int(max_size * 0.2))
    #     clean_cache_strategy = kwargs.get("clean_cache_strategy", "least_accessed_data")
    #     return SFDataManager(sqlite_path, index_path, dimension, top_k, max_size, clean_size, clean_cache_strategy)
    else:
        raise ValueError(f"Unsupported data manager: {data_manager_name}")


def _get_scalar_store(scalar_store: str, **kwargs):
    if scalar_store == "sqlite":
        sqlite_path = kwargs.get("sqlite_path", "sqlite.db")
        clean_cache_strategy = kwargs.get("clean_cache_strategy", "least_accessed_data")
        store = SQLite(sqlite_path, clean_cache_strategy)
    else:
        raise ValueError(f"Unsupported scalar store: {scalar_store}")
    return store


def _get_common_params(**kwargs):
    max_size = kwargs.get("max_size", 1000)
    clean_size = kwargs.get("clean_size", int(max_size * 0.2))
    top_k = kwargs.get("top_k", 1)
    dimension = kwargs.get("dimension", 0)
    if dimension <= 0:
        raise ValueError(f"the data manager should set the 'dimension' parameter greater than zero, "
                         f"current: {dimension}")
    return max_size, clean_size, dimension, top_k


# scalar_store + vector_store
def get_ss_data_manager(scalar_store: str, vector_store: str, **kwargs):
    max_size, clean_size, dimension, top_k = _get_common_params(**kwargs)
    scalar = _get_scalar_store(scalar_store, **kwargs)
    if vector_store == "milvus":
        vector = Milvus(collection_name="gpt_cache", dim=dimension, top_k=top_k, **kwargs)
    else:
        raise ValueError(f"Unsupported vector store: {vector_store}")
    return SSDataManager(max_size, clean_size, scalar, vector)


# scalar_store + vector_index
def get_si_data_manager(scalar_store: str, vector_index: str, **kwargs):
    max_size, clean_size, dimension, top_k = _get_common_params(**kwargs)
    store = _get_scalar_store(scalar_store, **kwargs)

    if vector_index == "faiss":
        index_path = kwargs.get("index_path", "faiss.index")
        index = Faiss(index_path, dimension, top_k)
    else:
        raise ValueError(f"Unsupported vector index: {vector_index}")

    return SIDataManager(max_size, clean_size, store, index)
