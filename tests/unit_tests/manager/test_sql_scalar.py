import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gptcache.manager.scalar_data.base import CacheData, Question
from gptcache.manager.scalar_data.sql_storage import SQLStorage
from gptcache.utils import import_sql_client


class TestSQLStore(unittest.TestCase):
    def test_sqlite(self):
        self._inner_test_normal("sqlite")
        self._inner_test_with_deps("sqlite")

    def test_duckdb(self):
        import_sql_client("duckdb")
        self._inner_test_normal("duckdb")
        self._inner_test_with_deps("duckdb")

    def _inner_test_normal(self, db_name: str):
        with TemporaryDirectory(dir="./") as root:
            db_path = Path(root) / f"{db_name}1.db"
            db = SQLStorage(
                db_type=db_name,
                url=f"{db_name}:///" + str(db_path),
                table_len_config={"question_question": 500},
            )
            db.create()
            data = []
            for i in range(1, 10):
                data.append(
                    CacheData(
                        "question_" + str(i),
                        ["answer_" + str(i)] * i,
                        np.random.rand(5),
                    )
                )

            db.batch_insert(data)
            data = db.get_data_by_id(1)
            self.assertEqual(data.question, "question_1")
            self.assertEqual(data.answers[0].answer, "answer_1")
            data = db.get_data_by_id(2)
            self.assertEqual(data.question, "question_2")
            self.assertEqual(data.answers[0].answer, "answer_2")
            self.assertEqual(data.answers[1].answer, "answer_2")
            q_id = db.batch_insert(
                [CacheData("question_single", "answer_singel", np.random.rand(5))]
            )[0]
            data = db.get_data_by_id(q_id)
            self.assertEqual(data.question, "question_single")
            self.assertEqual(data.answers[0].answer, "answer_singel")

            # test deleted
            self.assertEqual(len(db.get_ids(True)), 0)
            db.mark_deleted([1, 2, 3])
            self.assertEqual(db.get_ids(True), [1, 2, 3])
            self.assertEqual(db.count(is_all=True), 10)
            self.assertEqual(db.count(), 7)
            self.assertEqual(db.count_answers(), 46)
            db.clear_deleted_data()
            self.assertEqual(db.count_answers(), 40)
            self.assertEqual(db.count(is_all=True), 7)

    def _inner_test_with_deps(self, db_name: str):
        with TemporaryDirectory(dir="./") as root:
            db_path = Path(root) / f"{db_name}2.db"
            db = SQLStorage(db_type=db_name, url=f"{db_name}:///" + str(db_path))
            db.create()
            data_id = db.batch_insert(
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

            ret = db.get_data_by_id(data_id)
            self.assertEqual(ret.question.content, "test_question")
            self.assertEqual(ret.question.deps[0].name, "text")
            self.assertEqual(
                ret.question.deps[0].data, "how many people in this picture"
            )
            self.assertEqual(ret.question.deps[0].dep_type, 0)
            self.assertEqual(ret.question.deps[1].name, "image")
            self.assertEqual(ret.question.deps[1].data, "object_name")
            self.assertEqual(ret.question.deps[1].dep_type, 1)
