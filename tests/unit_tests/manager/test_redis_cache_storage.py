import time
import unittest

import numpy as np

from gptcache.manager.scalar_data.base import CacheData, Question
from gptcache.manager.scalar_data.redis_storage import RedisCacheStorage, get_models
from gptcache.utils import import_redis

import_redis()
from redis_om import get_redis_connection, RedisModel


class TestRedisStorage(unittest.TestCase):
    test_dbname = "gptcache_test"
    url = "redis://default:default@localhost:6379"

    def setUp(cls) -> None:
        cls._clear_test_db()

    @staticmethod
    def _clear_test_db():
        r = get_redis_connection(url=TestRedisStorage.url)
        r.flushall()
        r.flushdb()
        time.sleep(1)

    def test_normal(self):
        redis_storage = RedisCacheStorage(global_key_prefix=self.test_dbname,
                                          url=self.url)
        data = []
        for i in range(1, 10):
            data.append(
                CacheData(
                    "question_" + str(i),
                    ["answer_" + str(i)] * i,
                    np.random.rand(5).astype(np.float32)
                )
            )
        ids = redis_storage.batch_insert(data)

        for i, idx in enumerate(ids, start=1):
            result = redis_storage.get_data_by_id(idx)
            assert result.question == f"question_{i}"
            assert result.answers[0].answer == f"answer_{i}"

            assert all(np.equal(data[i - 1].embedding_data, result.embedding_data))

        q_id = redis_storage.batch_insert(
            [CacheData("question_single", "answer_single", np.random.rand(5))]
        )[0]
        data = redis_storage.get_data_by_id(q_id)
        assert data.question == "question_single"
        assert data.answers[0].answer == "answer_single"
        time.sleep(1)
        assert len(redis_storage.get_ids(True)) == 0
        redis_storage.mark_deleted(ids[:3])
        assert redis_storage.get_ids(True) == ids[:3]
        assert redis_storage.count(is_all=True) == 10
        assert redis_storage.count() == 7
        redis_storage.clear_deleted_data()

        assert redis_storage.count(is_all=True) == 7

    def test_with_deps(self):
        redis_storage = RedisCacheStorage(global_key_prefix=self.test_dbname,
                                          url=self.url)
        data_id = redis_storage.batch_insert(
            [
                CacheData(
                    Question.from_dict(
                        {
                            "content": "test_question",
                            "deps": [
                                {
                                    "name": "text",
                                    "data": "how many people in this picture",
                                    "dep_type": 0,
                                },
                                {
                                    "name": "image",
                                    "data": "object_name",
                                    "dep_type": 1,
                                },
                            ],
                        }
                    ),
                    "test_answer",
                    np.random.rand(5),
                )
            ]
        )[0]

        ret = redis_storage.get_data_by_id(data_id)
        assert ret.question.content == "test_question"
        assert ret.question.deps[0].name == "text"
        assert ret.question.deps[0].data == "how many people in this picture"
        assert ret.question.deps[0].dep_type == 0
        assert ret.question.deps[1].name == "image"
        assert ret.question.deps[1].data == "object_name"
        assert ret.question.deps[1].dep_type == 1

    def test_create_on(self):
        redis_storage = RedisCacheStorage(global_key_prefix=self.test_dbname,
                                          url=self.url)
        redis_storage.create()
        data = []
        for i in range(1, 10):
            data.append(
                CacheData(
                    "question_" + str(i),
                    ["answer_" + str(i)] * i,
                    np.random.rand(5),
                )
            )
        ids = redis_storage.batch_insert(data)
        data = redis_storage.get_data_by_id(ids[0])
        create_on1 = data.create_on
        last_access1 = data.last_access

        time.sleep(1)

        data = redis_storage.get_data_by_id(ids[0])
        create_on2 = data.create_on
        last_access2 = data.last_access

        assert create_on1 == create_on2
        assert last_access1 < last_access2

    def test_session(self):
        redis_storage = RedisCacheStorage(global_key_prefix=self.test_dbname,
                                          url=self.url)
        data = []
        for i in range(1, 11):
            data.append(
                CacheData(
                    "question_" + str(i),
                    ["answer_" + str(i)] * i,
                    np.random.rand(5),
                    session_id=str(1 if i <= 5 else 0)
                )
            )
        ids = redis_storage.batch_insert(data)
        assert len(redis_storage.list_sessions()) == 10
        assert len(redis_storage.list_sessions(session_id="0")) == 5
        assert len(redis_storage.list_sessions(session_id="1")) == 5
        assert len(redis_storage.list_sessions(session_id="1", key=ids[0])) == 1
        sessions = redis_storage.list_sessions(key=ids[0])
        assert len(sessions) == 1
        assert sessions[0].session_id == "1"

        redis_storage.delete_session(ids[:3])
        assert len(redis_storage.list_sessions()) == 7
