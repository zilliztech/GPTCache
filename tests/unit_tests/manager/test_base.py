import unittest

from gptcache.utils.error import NotFoundError
from gptcache.manager import CacheBase, VectorBase
from gptcache.manager.scalar_data.manager import CacheBase as InnerCacheBase
from gptcache.manager.vector_data.manager import VectorBase as InnerVectorBase


class TestBaseStore(unittest.TestCase):
    def test_cache_base(self):
        with self.assertRaises(EnvironmentError):
            InnerCacheBase()

        with self.assertRaises(NotFoundError):
            CacheBase("test_cache_base")

    def test_vector_base(self):
        with self.assertRaises(EnvironmentError):
            InnerVectorBase()

        with self.assertRaises(NotFoundError):
            VectorBase("test_cache_base")
