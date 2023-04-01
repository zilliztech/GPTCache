__all__ = ['Towhee']

from gptcache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.ranker.towhee')


def Towhee():
    return towhee.Towhee()
