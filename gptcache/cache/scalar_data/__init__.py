__all__ = ['SQLDataBase', 'SQL_URL']

from gptcache.util.lazy_import import LazyImport

sql_database = LazyImport('milvus', globals(), 'gptcache.cache.scalar_data.sqlalchemy')


def SQLDataBase(**kwargs):
    return sql_database.SQLDataBase(**kwargs)


SQL_URL = {
        'sqlite': 'sqlite:///./gpt_cache.db',
        'postgresql': 'postgresql+psycopg2://user:password@hostname:port/database_name',
        'mysql': 'mysql+pymysql://user:password@hostname:port/database_name',
        'mariadb': 'mariadb+pymysql://user:password@hostname:port/database_name',
        'sqlserver': 'mssql+pyodbc://user:password@database_name',
        'oracle': 'oracle+zxjdbc://user:password@hostname:port/database_name',
}
