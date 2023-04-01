__all__ = ['Towhee']

from gptcache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.encoder.towhee')


def Towhee(model="paraphrase-albert-small-v2"):
    return towhee.Towhee(model)
