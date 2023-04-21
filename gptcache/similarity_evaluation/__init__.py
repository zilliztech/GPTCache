from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation

__all__ = [
    "SimilarityEvaluation",
    "OnnxModelEvaluation",
    "NumpyNormEvaluation",
    "SearchDistanceEvaluation",
    "ExactMatchEvaluation",
    "KReciprocalEvaluation",
]

from gptcache.utils.lazy_import import LazyImport

onnx = LazyImport("onnx", globals(), "gptcache.similarity_evaluation.onnx")
np = LazyImport("np", globals(), "gptcache.similarity_evaluation.np")
distance = LazyImport("simple", globals(), "gptcache.similarity_evaluation.distance")
exact_match = LazyImport(
    "exact_match", globals(), "gptcache.similarity_evaluation.exact_match"
)
kreciprocal = LazyImport(
    "kreciprocal", globals(), "gptcache.similarity_evaluation.kreciprocal"
)


def OnnxModelEvaluation(model="GPTCache/albert-duplicate-onnx"):
    return onnx.OnnxModelEvaluation(model)


def NumpyNormEvaluation(enable_normal: bool = False, **kwargs):
    return np.NumpyNormEvaluation(enable_normal, **kwargs)


def SearchDistanceEvaluation(max_distance=4.0, positive=False):
    return distance.SearchDistanceEvaluation(max_distance, positive)


def ExactMatchEvaluation():
    return exact_match.ExactMatchEvaluation()


def KReciprocalEvaluation(vectordb, top_k=3, max_distance=4.0, positive=False):
    return kreciprocal.KReciprocalEvaluation(vectordb, top_k, max_distance, positive)
