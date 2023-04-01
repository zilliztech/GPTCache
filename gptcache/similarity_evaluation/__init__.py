__all__ = ['Towhee']

from gptcache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gptcache.similarity_evaluation.towhee')


def Towhee():
    return towhee.Towhee()
