
__all__ = ['OpenAI', 'Huggingface', 'SBERT', 'Cohere', 'Onnx', 'FastText']


from gptcache.utils.lazy_import import LazyImport

openai = LazyImport('openai', globals(), 'gptcache.embedding.openai')
huggingface = LazyImport('huggingface', globals(), 'gptcache.embedding.huggingface')
sbert = LazyImport('sbert', globals(), 'gptcache.embedding.sbert')
onnx = LazyImport('onnx', globals(), 'gptcache.embedding.onnx')
cohere = LazyImport('cohere', globals(), 'gptcache.embedding.cohere')
fasttext = LazyImport('fasttext', globals(), 'gptcache.embedding.fasttext')


def Cohere(model="large", api_key=None):
    return cohere.Cohere(model, api_key)

def OpenAI(model="text-embedding-ada-002", api_key=None):
    return openai.OpenAI(model, api_key)

def Huggingface(model="sentence-transformers/all-mpnet-base-v2"):
    return huggingface.Huggingface(model)

def SBERT(model="all-MiniLM-L6-v2"):
    return sbert.SBERT(model)

def Onnx(model="GPTCache/paraphrase-albert-onnx"):
    return onnx.Onnx(model)

def FastText(model="large", dim=None):
    return fasttext.FastText(model, dim)
