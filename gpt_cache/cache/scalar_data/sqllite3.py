from datetime import datetime
import sqlite3

import numpy as np


class SQLite:

    def __init__(self, db_path, clean_cache_strategy):
        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        create_tb_cmd = '''
            CREATE TABLE IF NOT EXISTS cache_data (
                id TEXT,
                data TEXT,
                embedding_data np_array,
                created_at DATETIME DEFAULT (DATETIME('now', 'localtime')),
                last_access_at DATETIME DEFAULT (DATETIME('now', 'localtime'))
            );'''
        self.cur.execute(create_tb_cmd)
        if clean_cache_strategy == "oldest_created_data":
            self.clean_cache_func = self.remove_oldest_created_data
        else:
            self.clean_cache_func = self.remove_least_accessed_data

    def count(self):
        res = self.cur.execute("SELECT COUNT(id) FROM cache_data")
        return res.fetchone()[0]

    def insert(self, key, data, embedding_data: np.ndarray):
        self.cur.execute("INSERT INTO cache_data (id, data, embedding_data) VALUES (?, ?, ?)",
                         (key, data, embedding_data))
        self.con.commit()

    def mult_insert(self, datas):
        self.cur.executemany("INSERT INTO cache_data (id, data, embedding_data) VALUES(?, ?, ?)", datas)
        self.con.commit()

    def select(self, key):
        res = self.cur.execute("SELECT data FROM cache_data WHERE id=?", (key, ))
        values = res.fetchone()
        # TODO batch asynchronous update
        last_read_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cur.execute("UPDATE cache_data SET last_access_at=? WHERE id=?", (last_read_time, key))
        self.con.commit()
        return values[0] if values is not None else None

    def select_all_embedding_data(self):
        self.cur.execute("select embedding_data from cache_data")
        rows = self.cur.fetchall()
        if len(rows) == 0:
            return None
        dimension = rows[0][0].shape[1]
        datas = np.empty((0, dimension)).astype('float32')
        for row in rows:
            datas = np.append(datas, row[0], axis=0)
        return datas

    def remove_oldest_created_data(self, count):
        self.cur.execute("""DELETE FROM cache_data
                            WHERE id IN (
                                SELECT id FROM cache_data ORDER BY created_at LIMIT ?
                            );""", (count,))
        self.con.commit()

    def remove_least_accessed_data(self, count):
        self.cur.execute("""DELETE FROM cache_data
                    WHERE id IN (
                        SELECT id FROM cache_data ORDER BY last_access_at LIMIT ?
                    );""", (count,))
        self.con.commit()

    def close(self):
        self.cur.close()
        self.con.close()
