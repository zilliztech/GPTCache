from .similarity_evaluation import SimilarityEvaluation


class SearchDistanceEvaluation(SimilarityEvaluation):

    def __init__(self, max_distance=4.0, positive=False):
        self.max_distance = max_distance
        self.positive = positive

    def evaluation(self, src_dict, cache_dict, **kwargs):
        distance, _ = cache_dict["search_result"]
        if distance < 0:
            distance = 0
        elif distance > self.max_distance:
            distance = self.max_distance
        if self.positive:
            return distance
        return self.max_distance - distance

    def range(self):
        return 0.0, self.max_distance
