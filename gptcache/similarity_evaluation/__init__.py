__all__ = ['Towhee', 'Onnx']

from gptcache.utils.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.similarity_evaluation.towhee')
onnx = LazyImport('onnx', globals(), 'gptcache.similarity_evaluation.onnx')


def Towhee():
    return towhee.Towhee()

def Onnx(model = 'GPTCache/albert-duplicate-onnx'):
    return onnx.Onnx(model)
