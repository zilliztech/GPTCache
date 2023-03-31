__all__ = ['Towhee']

from gpt_cache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gpt_cache.embedding.towhee')


def Towhee():
    return towhee.Towhee()
