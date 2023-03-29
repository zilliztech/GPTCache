def absolute_evaluation(src_embedding_data, cache_data, **kwargs):
    if not isinstance(cache_data, tuple):
        return 0
    return 100 if cache_data[0] == src_embedding_data else 0
