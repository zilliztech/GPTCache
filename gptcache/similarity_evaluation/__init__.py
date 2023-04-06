__all__ = ['OnnxModelEvaluation', 'NumpyNormEvaluation', 'SearchDistanceEvaluation', 'ExactMatchEvaluation']

from gptcache.utils.lazy_import import LazyImport

onnx = LazyImport('onnx', globals(), 'gptcache.similarity_evaluation.onnx')
np = LazyImport('np', globals(), 'gptcache.similarity_evaluation.np')
distance = LazyImport('simple', globals(), 'gptcache.similarity_evaluation.distance')
exact_match = LazyImport('exact_match', globals(), 'gptcache.similarity_evaluation.exact_match')

def OnnxModelEvaluation(model = 'GPTCache/albert-duplicate-onnx'):
    return onnx.OnnxModelEvaluation(model)

def NumpyNormEvaluation(enable_normal: bool = False):
    return np.NumpyNormEvaluation(enable_normal)

def SearchDistanceEvaluation(max_distance = 4.0, positive = False):
    return distance.SearchDistanceEvaluation(max_distance, positive)

def ExactMatchEvaluation():
    return exact_match.ExactMatchEvaluation()
