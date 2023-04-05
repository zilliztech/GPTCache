__all__ = ['SQLDataBase', 'SQL_URL']

from gptcache.utils.lazy_import import LazyImport

sql_database = LazyImport('milvus', globals(), 'gptcache.cache.scalar_data.sqlalchemy')


def SQLDataBase(**kwargs):
    return sql_database.SQLDataBase(**kwargs)


SQL_URL = {
        'sqlite': 'sqlite:///./sqlite.db',
        'postgresql': 'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres',
        'mysql': 'mysql+pymysql://root:123456@127.0.0.1:3306/mysql',
        'mariadb': 'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql',
        'sqlserver': 'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server',
        'oracle': 'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8',
}
