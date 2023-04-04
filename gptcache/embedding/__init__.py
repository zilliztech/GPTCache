__all__ = ['Towhee']

from gptcache.utils.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.embedding.towhee')


def Towhee(model="paraphrase-albert-small-v2"):
    return towhee.Towhee(model)
