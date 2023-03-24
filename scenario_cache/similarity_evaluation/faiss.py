def faiss_evaluation(src_embedding_data, cache_data, **kwargs) -> int:
    distance, vector_data = cache_data
    print("distance", distance)
    return distance
