from gpt_cache.cache.data_manager import DataManager


def get_data_manager(name: str, **kwargs) -> DataManager:
    if name == "map":
        from gpt_cache.cache.data_manager import MapDataManager

        return MapDataManager(kwargs.get("data_path", "data_map.txt"),
                              kwargs.get("max_size", 100))
    elif name == "sqlite_faiss":
        from gpt_cache.cache.data_manager import SFDataManager

        dimension = kwargs.get("dimension", 0)
        if dimension <= 0:
            raise ValueError(f"the sqlite_faiss data manager should set the 'dimension' parameter greater than zero, "
                             f"current: {dimension}")
        top_k = kwargs.get("top_k", 1)
        sqlite_path = kwargs.get("sqlite_path", "sqlite.db")
        index_path = kwargs.get("index_path", "faiss.index")
        max_size = kwargs.get("max_size", 1000)
        clean_size = kwargs.get("clean_size", int(max_size * 0.2))
        clean_cache_strategy = kwargs.get("clean_cache_strategy", "least_accessed_data")
        return SFDataManager(sqlite_path, index_path, dimension, top_k, max_size, clean_size, clean_cache_strategy)
    else:
        raise ValueError(f"Unsupported data manager: {name}")
