from .similarity_evaluation import SimilarityEvaluation


class ExactMatchEvaluation(SimilarityEvaluation):

    def evaluation(self, src_dict, cache_dict, **kwargs):
        return 1 if cache_dict["question"] == src_dict["question"] else 0

    def range(self):
        return 0, 1
