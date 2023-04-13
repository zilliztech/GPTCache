import unittest

from gptcache.utils.error import NotFoundStoreError
from gptcache.manager import CacheBase, VectorBase
from gptcache.manager.scalar_data.manager import CacheBase as InnerCacheBase
from gptcache.manager.vector_data.manager import VectorBase as InnerVectorBase


class TestBaseStore(unittest.TestCase):
    def test_cache_base(self):
        with self.assertRaises(EnvironmentError):
            InnerCacheBase()

        with self.assertRaises(NotFoundStoreError):
            CacheBase("test_cache_base")

    def test_vector_base(self):
        with self.assertRaises(EnvironmentError):
            InnerVectorBase()

        with self.assertRaises(NotFoundStoreError):
            VectorBase("test_cache_base")
