__all__ = ['Towhee']

from gpt_cache.util.lazy_import import LazyImport

towhee = LazyImport('towhee', globals(), 'gpt_cache.similarity_evaluation.towhee')


def Towhee():
    return towhee.Towhee()
