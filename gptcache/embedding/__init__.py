__all__ = ['Towhee', 'OpenAI', 'Huggingface', 'SBERT', 'Cohere']

from gptcache.utils.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.embedding.towhee')
openai = LazyImport('openai', globals(), 'gptcache.embedding.openai')
huggingface = LazyImport('huggingface', globals(), 'gptcache.embedding.huggingface')
sbert = LazyImport('sbert', globals(), 'gptcache.embedding.sbert')
cohere = LazyImport('cohere', globals(), 'gptcache.embedding.cohere')


def Towhee(model="paraphrase-albert-small-v2-onnx"):
    return towhee.Towhee(model)

def Cohere(model="large", api_key=None):
    return cohere.Cohere(model, api_key)

def OpenAI(model="text-embedding-ada-002", api_key=None):
    return openai.OpenAI(model, api_key)

def Huggingface(model="sentence-transformers/all-mpnet-base-v2"):
    return huggingface.Huggingface(model)

def SBERT(model="all-MiniLM-L6-v2"):
    return sbert.SBERT(model)