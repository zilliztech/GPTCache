__all__ = ['Towhee']

from gptcache.utils.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.embedding.towhee')
openai = LazyImport('openai', globals(), 'gptcache.embedding.openai')


def Towhee(model="paraphrase-albert-small-v2-onnx"):
    return towhee.Towhee(model)

def OpenAI(api_key, model="text-embedding-ada-002"):
    return openai.OpenAI(api_key, model)