import sqlite3


class SQLite:

    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        create_tb_cmd = '''
            CREATE TABLE IF NOT EXISTS cache_data
            (id TEXT PRIMARY KEY,
            data TEXT);
            '''
        self.cur.execute(create_tb_cmd)

    def insert(self, key, data):
        self.cur.execute("INSERT INTO cache_data VALUES(?, ?)", (key, data))
        self.con.commit()

    # datas format
    # datas = [
    #     ("1", "Monty Python Live at the Hollywood Bowl"),
    #     ("2", "Monty Python's The Meaning of Life"),
    # ]
    def mult_insert(self, datas):
        self.cur.executemany("INSERT INTO cache_data VALUES(?, ?)", datas)
        self.con.commit()

    def select(self, key):
        res = self.cur.execute("SELECT data FROM cache_data WHERE id=?", (key, ))
        values = res.fetchone()
        return values[0] if values is not None else None

    def close(self):
        self.cur.close()
        self.con.close()
