__all__ = ['Towhee']

from gpt_cache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gpt_cache.embedding.towhee')


def Towhee(model="paraphrase-albert-small-v2"):
    return towhee.Towhee(model)
