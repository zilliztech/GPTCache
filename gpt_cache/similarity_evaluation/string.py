def absolute_evaluation(src_dict, cache_dict, **kwargs):
    return 100 if cache_dict["question"] == src_dict["question"] else 0
