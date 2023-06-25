import unittest

import gptcache
from gptcache.manager.eviction.manager import EvictionBase
from gptcache.utils.error import NotFoundError


class TestEviction(unittest.TestCase):
    def test_memory_cache_eviction(self):
        with self.assertRaises(EnvironmentError):
            EvictionBase()

        with self.assertRaises(NotFoundError):
            EvictionBase.get(
                name="test_eviction",
                policy="lru",
                maxsize=10,
                clean_size=2,
                on_evict=lambda x: print(x),
            )

        with self.assertRaises(ValueError):
            EvictionBase.get(
                name="memory",
                policy="test_eviction",
                maxsize=10,
                clean_size=2,
                on_evict=lambda x: print(x),
            )

    def test_lru(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                datas.remove(delete)
            return

        eviction_base = EvictionBase.get(
            name="memory", policy="lru", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction_base.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        eviction_base.get(1)
        add_data(5)
        self.assertEqual(3, len(datas))
        self.assertTrue(datas.index(1) != -1)

    def test_lru_no_clean_size(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                datas.remove(delete)
            return

        eviction_base = gptcache.manager.eviction.EvictionBase(
            name="memory", policy="lru", maxsize=5, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction_base.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        add_data(5)
        eviction_base.get(1)
        add_data(6)
        self.assertEqual(5, len(datas))
        self.assertTrue(datas.index(1) != -1)

    def test_lfu(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                datas.remove(delete)
            return

        eviction_base = EvictionBase.get(
            name="memory", policy="lfu", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction_base.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        eviction_base.get(2)
        eviction_base.get(2)
        eviction_base.get(3)
        eviction_base.get(3)
        eviction_base.get(4)
        eviction_base.get(4)
        eviction_base.get(1)
        add_data(5)
        self.assertEqual(3, len(datas))
        self.assertFalse(1 in datas)

    def test_fifo(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                datas.remove(delete)
            return

        eviction_base = EvictionBase.get(
            name="memory", policy="fifo", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction_base.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        eviction_base.get(1)
        add_data(5)
        self.assertEqual(3, len(datas))
        self.assertFalse(1 in datas)

    def test_rr(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                datas.remove(delete)
            return

        eviction_base = EvictionBase.get(
            name="memory", policy="rr", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction_base.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        eviction_base.get(1)
        add_data(5)
        self.assertEqual(3, len(datas))
