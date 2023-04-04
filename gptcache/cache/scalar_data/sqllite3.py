from datetime import datetime
from typing import Iterable, Tuple
import sqlite3

import numpy as np

from .scalar_store import ScalarStore


class SQLite(ScalarStore):

    def __init__(self, db_path: str, eviction_strategy: str):
        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        create_tb_cmd = '''
            CREATE TABLE IF NOT EXISTS cache_data (
                id TEXT,
                question TEXT,
                answer TEXT,
                embedding_data np_array,
                created_at DATETIME DEFAULT (DATETIME('now', 'localtime')),
                last_access_at DATETIME DEFAULT (DATETIME('now', 'localtime'))
            );'''
        self.cur.execute(create_tb_cmd)
        if eviction_strategy == "oldest_created_data":
            self.get_eviction_data_id = self.get_oldest_created_data
        else:
            self.get_eviction_data_id = self.get_least_accessed_data

    def init(self, **kwargs):
        pass

    def count(self):
        res = self.cur.execute("SELECT COUNT(id) FROM cache_data")
        return res.fetchone()[0]

    def insert(self, key: str, question: str, answer: str, embedding_data: np.ndarray):
        self.cur.execute("INSERT INTO cache_data (id, question, answer, embedding_data) VALUES (?, ?, ?, ?)",
                         (key, question, answer, embedding_data))
        self.con.commit()

    def mult_insert(self, datas: Iterable[Tuple[str, str, str, np.ndarray]]):
        self.cur.executemany("INSERT INTO cache_data (id, question, answer, embedding_data) VALUES(?, ?, ?, ?)", datas)
        self.con.commit()

    def select_data(self, key: str):
        res = self.cur.execute("SELECT question, answer FROM cache_data WHERE id=?", (key, ))
        values = res.fetchone()
        # TODO batch asynchronous update
        last_read_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cur.execute("UPDATE cache_data SET last_access_at=? WHERE id=?", (last_read_time, key))
        self.con.commit()
        return (values[0], values[1]) if values is not None else None

    def select_all_embedding_data(self):
        self.cur.execute("select embedding_data from cache_data")
        rows = self.cur.fetchall()
        if len(rows) == 0:
            return None
        dimension = rows[0][0].reshape(1, -1).shape[1]
        datas = np.empty((0, dimension)).astype('float32')
        for row in rows:
            datas = np.append(datas, row[0].reshape(1, -1), axis=0)
        return datas

    def get_oldest_created_data(self, count: int):
        return self.cur.execute("SELECT id FROM cache_data ORDER BY created_at LIMIT ?", (count, ))

    def get_least_accessed_data(self, count: int):
        return self.cur.execute("SELECT id FROM cache_data ORDER BY last_access_at LIMIT ?", (count, ))

    def eviction(self, count: int):
        res = self.get_eviction_data_id(count)
        ids = []
        for row in res.fetchall():
            ids.append(row[0])
        delete_sql = "DELETE FROM cache_data WHERE id IN ({})".format(','.join('?' * len(ids)))
        self.cur.execute(delete_sql, ids)
        self.con.commit()
        return ids

    def close(self):
        self.cur.close()
        self.con.close()
