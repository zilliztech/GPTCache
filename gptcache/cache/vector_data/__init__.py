__all__ = ['Milvus', 'Faiss', 'Chromadb']

from gptcache.util.lazy_import import LazyImport

milvus = LazyImport('milvus', globals(), 'gptcache.cache.vector_data.milvus')
faiss = LazyImport('faiss', globals(), 'gptcache.cache.vector_data.faiss')
chromadb = LazyImport('chromadb', globals(), 'gptcache.cache.vector_data.chromadb')


def Milvus(**kwargs):
    return milvus.Milvus(**kwargs)


def Faiss(index_file_path, dimension, top_k, skip_file=False):
    return faiss.Faiss(index_file_path, dimension, top_k, skip_file)


def Chromadb(**kwargs):
    return chromadb.Chromadb(**kwargs)
