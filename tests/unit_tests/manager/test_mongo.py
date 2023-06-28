import time
import unittest

import numpy as np
from mongoengine import connect, disconnect

from gptcache.manager.scalar_data.base import CacheData, Question
from gptcache.manager.scalar_data.mongo import MongoStorage


class TestMongoStore(unittest.TestCase):
    def test_mongo(self):
        test_dbname = "gptcache_test"
        self._clear_test_db(test_dbname)
        self._inner_test_normal(test_dbname)

        self._clear_test_db(test_dbname)
        self._inner_test_with_deps(test_dbname)

        self._clear_test_db(test_dbname)
        self._test_create_on(dbname=test_dbname)

        self._clear_test_db(test_dbname)
        self._test_session(dbname=test_dbname)

    def _clear_test_db(self, dbname):
        con = connect(db=dbname)
        con.drop_database(dbname)
        disconnect()

    def _inner_test_normal(self, dbname: str):
        mongo_storage = MongoStorage(dbname=dbname)
        data = []
        for i in range(1, 10):
            data.append(
                CacheData(
                    "question_" + str(i),
                    ["answer_" + str(i)] * i,
                    np.random.rand(5)
                )
            )
        mongo_storage.batch_insert(data)

        for i in range(1, 10):
            data = mongo_storage.get_data_by_id(i)
            self.assertEqual(data.question, f"question_{i}")
            self.assertEqual(data.answers[0].answer, f"answer_{i}")

        q_id = mongo_storage.batch_insert(
            [CacheData("question_single", "answer_single", np.random.rand(5))]
        )[0]
        data = mongo_storage.get_data_by_id(q_id)
        self.assertEqual(data.question, "question_single")
        self.assertEqual(data.answers[0].answer, "answer_single")

        self.assertEqual(len(mongo_storage.get_ids(True)), 0)
        mongo_storage.mark_deleted([1, 2, 3])
        self.assertEqual(mongo_storage.get_ids(True), [1, 2, 3])
        self.assertEqual(mongo_storage.count(is_all=True), 10)
        self.assertEqual(mongo_storage.count(), 7)
        self.assertEqual(mongo_storage.count_answers(), 46)
        mongo_storage.clear_deleted_data()
        self.assertEqual(mongo_storage.count_answers(), 40)
        self.assertEqual(mongo_storage.count(is_all=True), 7)

    def _inner_test_with_deps(self, dbname: str):
        mongo_storage = MongoStorage(dbname=dbname)
        data_id = mongo_storage.batch_insert(
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

        ret = mongo_storage.get_data_by_id(data_id)
        self.assertEqual(ret.question.content, "test_question")
        self.assertEqual(ret.question.deps[0].name, "text")
        self.assertEqual(
            ret.question.deps[0].data, "how many people in this picture"
        )
        self.assertEqual(ret.question.deps[0].dep_type, 0)
        self.assertEqual(ret.question.deps[1].name, "image")
        self.assertEqual(ret.question.deps[1].data, "object_name")
        self.assertEqual(ret.question.deps[1].dep_type, 1)

    def _test_create_on(self, dbname):
        mongo_storage = MongoStorage(dbname=dbname)
        mongo_storage.create()
        data = []
        for i in range(1, 10):
            data.append(
                CacheData(
                    "question_" + str(i),
                    ["answer_" + str(i)] * i,
                    np.random.rand(5),
                )
            )
        mongo_storage.batch_insert(data)
        data = mongo_storage.get_data_by_id(1)
        create_on1 = data.create_on
        last_access1 = data.last_access

        time.sleep(1)

        data = mongo_storage.get_data_by_id(1)
        create_on2 = data.create_on
        last_access2 = data.last_access

        assert create_on1 == create_on2
        assert last_access1 < last_access2

    def _test_session(self, dbname):
        mongo_storage = MongoStorage(dbname=dbname)
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
        mongo_storage.batch_insert(data)
        self.assertEqual(len(mongo_storage.list_sessions()), 10)
        self.assertEqual(len(mongo_storage.list_sessions(session_id="0")), 5)
        self.assertEqual(len(mongo_storage.list_sessions(session_id="1")), 5)
        self.assertEqual(len(mongo_storage.list_sessions(session_id="1", key=1)), 1)

        mongo_storage.delete_session([1, 2, 3])
        self.assertEqual(len(mongo_storage.list_sessions()), 7)
