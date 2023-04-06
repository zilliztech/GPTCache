__all__ = ['Onnx']

from gptcache.utils.lazy_import import LazyImport

onnx = LazyImport('onnx', globals(), 'gptcache.similarity_evaluation.onnx')

def Onnx(model = 'GPTCache/albert-duplicate-onnx'):
    return onnx.Onnx(model)
