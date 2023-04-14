import unittest
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory

from gptcache.manager.scalar_data.sql_storage import SQLStorage
from gptcache.manager.scalar_data.base import CacheData


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
            self.assertEqual(db.get_data_by_id(1)[:2], ['question_1', 'answer_1'])
            self.assertEqual(db.get_data_by_id(2)[:2], ['question_2', ['answer_2', 'answer_2']])

            q_id = db.batch_insert([CacheData('question_single', 'answer_singel', np.random.rand(5))])[0]
            self.assertEqual(db.get_data_by_id(q_id)[:2], ['question_single', 'answer_singel'])

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

            # test access
            # 4 -> 10
            self.assertEqual(db.get_old_access(10), [4, 5, 6, 7, 8, 9, 10])
            db.update_access_time(4)
            # 5 -> 10, 4
            self.assertEqual(db.get_old_access(10), [5, 6, 7, 8, 9, 10, 4])

            # test get_old_create
            self.assertEqual(db.get_old_create(10), [4, 5, 6, 7, 8, 9, 10])

