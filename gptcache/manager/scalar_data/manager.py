from gptcache.utils import import_sql_client
from gptcache.utils.error import NotFoundError

SQL_URL = {
    "sqlite": "sqlite:///./sqlite.db",
    "duckdb": "duckdb:///./duck.db",
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

    Generate specific CacheStorage with the configuration. For example, setting for
       `SQLDataBase` (with `name`, `sql_url` and `table_name` params) to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.

    :param name: the name of the cache storage, it is support 'sqlite', 'postgresql', 'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type name: str
    :param sql_url: the url of the sql database for cache, such as '<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>',
        and the default value is related to the `cache_store` parameter,

        - 'sqlite:///./sqlite.db' for 'sqlite',
        - 'duckdb:///./duck.db' for 'duckdb',
        - 'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres' for 'postgresql',
        - 'mysql+pymysql://root:123456@127.0.0.1:3306/mysql' for 'mysql',
        - 'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql' for 'mariadb',
        - 'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server' for 'sqlserver',
        - 'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8' for 'oracle'.
    :type sql_url: str
    :param table_name: the table name for sql database, defaults to 'gptcache'.
    :type table_name: str
    :param table_len_config: the table length config for sql database, defaults to {}. the key includes:

        - 'question_question': the question column size in the question table, default to 3000.
        - 'answer_answer': the answer column size in the answer table, default to 3000.
        - 'session_id': the session id column size in the session table, default to 1000.
        - 'dep_name': the name column size in the dep table, default to 1000.
        - 'dep_data': the data column size in the dep table, default to 3000.
    :type table_len_config: dict

    :return: CacheStorage.

    Example:
        .. code-block:: python

            from gptcache.manager import CacheBase

            cache_base = CacheBase('sqlite')
    """

    def __init__(self):
        raise EnvironmentError(
            "CacheBase is designed to be instantiated, please using the `CacheBase.get(name)`."
        )

    @staticmethod
    def get(name, **kwargs):
        if name in [
            "sqlite",
            "duckdb",
            "postgresql",
            "mysql",
            "mariadb",
            "sqlserver",
            "oracle",
        ]:
            from gptcache.manager.scalar_data.sql_storage import SQLStorage

            sql_url = kwargs.get("sql_url", SQL_URL[name])
            table_name = kwargs.get("table_name", TABLE_NAME)
            table_len_config = kwargs.get("table_len_config", {})
            import_sql_client(name)
            cache_base = SQLStorage(
                db_type=name,
                url=sql_url,
                table_name=table_name,
                table_len_config=table_len_config,
            )
        elif name == "mongo":
            from gptcache.manager.scalar_data.mongo import MongoStorage

            return MongoStorage(
                host=kwargs.get("mongo_host", "localhost"),
                port=kwargs.get("mongo_port", 27017),
                dbname=kwargs.get("dbname", TABLE_NAME),
                username=kwargs.get("username"),
                password=kwargs.get("password")
            )
        elif name == "redis":
            from gptcache.manager.scalar_data.redis_storage import RedisCacheStorage

            return RedisCacheStorage(
                host=kwargs.pop("redis_host", "localhost"),
                port=kwargs.pop("redis_port", 6379),
                global_key_prefix=kwargs.pop("global_key_prefix", TABLE_NAME),
                **kwargs
            )
        else:
            raise NotFoundError("cache store", name)
        return cache_base
