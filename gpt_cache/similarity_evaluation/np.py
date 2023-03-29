import numpy as np


def linalg_norm_evaluation(src_embedding_data, cache_data, **kwargs):
    return np.linalg.norm(src_embedding_data - cache_data)
