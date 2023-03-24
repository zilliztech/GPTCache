import numpy as np


def linalg_norm_evaluation(src_embedding_data, cache_data, **kwargs) -> int:
    return int(np.linalg.norm(src_embedding_data - cache_data) * 100)
