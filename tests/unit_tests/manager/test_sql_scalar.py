import unittest
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory

from gptcache.manager.scalar_data.sql_storage import SQLStorage
from gptcache.manager.scalar_data.base import CacheData, Question


class TestSQLStore(unittest.TestCase):
    def test_normal(self):
        with TemporaryDirectory(dir='./') as root:
            db_path = Path(root) / 'sqlite.db'
            db = SQLStorage(url="sqlite:///" + str(db_path))
            db.create()
            data = []
            for i in range(1, 10):
                data.append(CacheData('question_' + str(i), ['answer_' + str(i)] * i, np.random.rand(5)))

            db.batch_insert(data)
            data = db.get_data_by_id(1)
            self.assertEqual(data.question, 'question_1')
            self.assertEqual(data.answers[0].answer, 'answer_1')
            data = db.get_data_by_id(2)
            self.assertEqual(data.question, 'question_2')
            self.assertEqual(data.answers[0].answer, 'answer_2')
            self.assertEqual(data.answers[1].answer, 'answer_2')
            q_id = db.batch_insert([CacheData('question_single', 'answer_singel', np.random.rand(5))])[0]
            data = db.get_data_by_id(q_id)
            self.assertEqual(data.question, 'question_single')
            self.assertEqual(data.answers[0].answer, 'answer_singel')

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

    def test_with_deps(self):
        with TemporaryDirectory(dir='./') as root:
            db_path = Path(root) / 'sqlite.db'
            db = SQLStorage(url="sqlite:///" + str(db_path))
            db.create()        
            data_id = db.batch_insert([
                CacheData(
                    Question.from_dict({
                        "content": "test_question",
                        "deps": [
                            {"name": "text", "data": "how many people in this picture", "dep_type": 0},
                            {"name": "image", "data": "object_name", "dep_type": 1}
                        ]
                    }),
                    'test_answer', np.random.rand(5))
            ])[0]

            ret = db.get_data_by_id(data_id)
            self.assertEqual(ret.question.content, "test_question")
            self.assertEqual(ret.question.deps[0].name, "text")
            self.assertEqual(ret.question.deps[0].data, "how many people in this picture")
            self.assertEqual(ret.question.deps[0].dep_type, 0)
            self.assertEqual(ret.question.deps[1].name, "image")
            self.assertEqual(ret.question.deps[1].data, "object_name")
            self.assertEqual(ret.question.deps[1].dep_type, 1)
