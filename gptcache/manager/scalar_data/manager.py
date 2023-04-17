from gptcache.utils import import_sql_client
from gptcache.utils.error import NotFoundError


SQL_URL = {
    "sqlite": "sqlite:///./sqlite.db",
    "postgresql": "postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres",
    "mysql": "mysql+pymysql://root:123456@127.0.0.1:3306/mysql",
    "mariadb": "mariadb+pymysql://root:123456@127.0.0.1:3307/mysql",
    "sqlserver": "mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server",
    "oracle": "oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8",
}
TABLE_NAME = "gptcache"


# pylint: disable=import-outside-toplevel
class CacheBase:
    """
    CacheBase to manager the cache storage.
    """

    def __init__(self):
        raise EnvironmentError(
            "CacheBase is designed to be instantiated, please using the `CacheBase.get(name)`."
        )

    @staticmethod
    def get(name, **kwargs):
        if name in ["sqlite", "postgresql", "mysql", "mariadb", "sqlserver", "oracle"]:
            from gptcache.manager.scalar_data.sql_storage import SQLStorage

            sql_url = kwargs.get("sql_url", SQL_URL[name])
            table_name = kwargs.get("table_name", TABLE_NAME)
            import_sql_client(name)
            cache_base = SQLStorage(db_type=name, url=sql_url, table_name=table_name)
        else:
            raise NotFoundError("cache store", name)
        return cache_base
