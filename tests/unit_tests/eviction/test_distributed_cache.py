import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from redis_om import get_redis_connection

from gptcache.embedding import Onnx
from gptcache.manager import manager_factory, get_data_manager, CacheBase, VectorBase
from gptcache.manager.eviction import EvictionBase


class TestDistributedCache(unittest.TestCase):
    url: str = "redis://default:default@localhost:6379"

    def setUp(cls) -> None:
        cls._clear_test_db()

    @staticmethod
    def _clear_test_db():
        r = get_redis_connection(url=TestDistributedCache.url)
        r.flushall()
        r.flushdb()
        time.sleep(1)

    def test_redis_only_cache_eviction(self):
        manager = manager_factory("redis,faiss",
                                  eviction_manager="redis",
                                  vector_params={"dimension": 5},
                                  eviction_params=dict(url=self.url,
                                                       maxmemory="100mb",
                                                       policy="allkeys-lru"))
        self.assertEqual(type(manager.eviction_base).__name__, "NoOpEviction")
        self.assertEqual(type(manager.s).__name__, "RedisCacheStorage")

    def test_redis_sqlite_cache_eviction(self):
        with TemporaryDirectory(dir="./") as root:
            db_name = "sqlite"
            db_path = Path(root) / f"{db_name}.db"
            manager = manager_factory("sqlite,faiss",
                                      eviction_manager="redis",
                                      scalar_params={
                                          "url": f"{db_name}:///" + str(db_path),
                                      },
                                      vector_params={"dimension": 5},
                                      eviction_params=dict(url=self.url,
                                                           maxmemory="100mb",
                                                           policy="allkeys-lru"))

            self.assertEqual(type(manager.s).__name__, "SQLStorage")
            self.assertEqual(type(manager.eviction_base).__name__, "RedisCacheEviction")
            self.assertEqual(manager.eviction_base.policy, "allkeys-lru")
            self.assertEqual(manager.eviction_base._ttl, None)

    def test_lru_cache(self):
        onnx = Onnx()
        data_manager = manager_factory("redis,faiss",
                                       eviction_manager="redis",
                                       scalar_params={"url": self.url,
                                                      "maxmemory": "1800kb",
                                                      "policy": "allkeys-lru"
                                                      },
                                       vector_params={"dimension": onnx.dimension},
                                       eviction_params=dict(url=self.url)
                                       )
        questions = []
        answers = []
        idx_list = []
        for i in range(50):
            idx_list.append(i)
            questions.append(f'This is a question_{i}')
            answers.append(f'This is an answer_{i}')

        for idx, question, answer in zip(idx_list, questions, answers):
            embedding = onnx.to_embeddings(question)
            data_manager.save(
                question=question,
                answer=answer,
                embedding_data=embedding
            )

        self.assertNotEquals(data_manager.s.count(), len(idx_list))

    def test_noeviction_policy(self):
        onnx = Onnx()
        data_manager = manager_factory("redis,faiss",
                                       eviction_manager="redis",
                                       scalar_params={"url": self.url
                                                      },
                                       vector_params={"dimension": onnx.dimension},
                                       eviction_params={"maxmemory": "0",
                                                        "policy": "noeviction"}
                                       )
        questions = []
        answers = []
        idx_list = []
        for i in range(100):
            idx_list.append(i)
            questions.append(f'This is a question_{i}')
            answers.append(f'This is an answer_{i}')

        for idx, question, answer in zip(idx_list, questions, answers):
            embedding = onnx.to_embeddings(question)
            data_manager.save(
                question=question,
                answer=answer,
                embedding_data=embedding
            )

        self.assertEqual(data_manager.s.count(), len(idx_list))

    def test_ttl(self):
        onnx = Onnx()
        data_manager = manager_factory("redis,faiss",
                                       eviction_manager="redis",
                                       scalar_params={"url": self.url
                                                      },
                                       vector_params={"dimension": onnx.dimension},
                                       eviction_params={"maxmemory": "0",
                                                        "policy": "noeviction",
                                                        "ttl": 1}
                                       )
        questions = []
        answers = []
        idx_list = []
        embeddings = []
        for i in range(10):
            idx_list.append(i)
            questions.append(f'This is a question_{i}')
            answers.append(f'This is an answer_{i}')
            embeddings.append(onnx.to_embeddings(questions[-1]))

        data_manager.import_data(questions, answers, embedding_datas=embeddings,
                                 session_ids=[None for _ in range(len(questions))])
        time.sleep(5)
        self.assertEqual(data_manager.s.count(), 0)

    def test_redis_only_config(self):
        onnx = Onnx()
        data_manager = get_data_manager(cache_base=CacheBase("redis", maxmemory="100mb", policy="allkeys-lru"),
                                        vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                        eviction_base=EvictionBase("redis"))
        questions = []
        answers = []
        idx_list = []
        embeddings = []
        for i in range(10):
            idx_list.append(i)
            questions.append(f'This is a question_{i}')
            answers.append(f'This is an answer_{i}')
            embeddings.append(onnx.to_embeddings(questions[-1]))

        data_manager.import_data(questions, answers, embedding_datas=embeddings,
                                 session_ids=[None for _ in range(len(questions))])
        search_data = data_manager.search(embeddings[0], top_k=1)
        for res in search_data:
            self.assertEqual(data_manager.eviction_base.get(res[1]), "True")

    def test_redis_only_with_no_op_eviction_config(self):
        onnx = Onnx()
        data_manager = get_data_manager(cache_base=CacheBase("redis", maxmemory="100mb", policy="allkeys-lru"),
                                        vector_base=VectorBase("faiss", dimension=onnx.dimension),
                                        eviction_base=EvictionBase("no_op_eviction"))
        questions = []
        answers = []
        idx_list = []
        embeddings = []
        for i in range(10):
            idx_list.append(i)
            questions.append(f'This is a question_{i}')
            answers.append(f'This is an answer_{i}')
            embeddings.append(onnx.to_embeddings(questions[-1]))

        data_manager.import_data(questions, answers, embedding_datas=embeddings,
                                 session_ids=[None for _ in range(len(questions))])
        search_data = data_manager.search(embeddings[0], top_k=1)
        for res in search_data:
            self.assertEqual(data_manager.eviction_base.get(res[1]), None)
