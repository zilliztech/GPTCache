import numpy as np


def linalg_norm_evaluation(src_dict, cache_dict, **kwargs):
    src_embedding = src_dict["embedding"]
    _, cache_embedding = cache_dict["search_result"]
    return np.linalg.norm(src_embedding - cache_embedding)
