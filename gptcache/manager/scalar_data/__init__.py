__all__ = ["CacheBase"]

from gptcache.utils.lazy_import import LazyImport

scalar_manager = LazyImport(
    "scalar_manager", globals(), "gptcache.manager.scalar_data.manager"
)


def CacheBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration. For example, setting for
       `SQLDataBase` (with `name`, `sql_url` and `table_name` params) to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.

    :param name: the name of the cache storage, it is support 'sqlite', 'postgresql', 'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type name: str
    :param sql_url: the url of the sql database for cache, such as '<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>',
                    and the default value is related to the `cache_store` parameter, 'sqlite:///./sqlite.db' for 'sqlite',
                    'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres' for 'postgresql',
                    'mysql+pymysql://root:123456@127.0.0.1:3306/mysql' for 'mysql',
                    'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql' for 'mariadb',
                    'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server' for 'sqlserver',
                    'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8' for 'oracle'.
    :type sql_url: str
    :param table_name: the table name for sql database, defaults to 'gptcache'.
    :type table_name: str

    :return: CacheStorage.

    Example:
        .. code-block:: python

            from gptcache.manager import CacheBase

            cache_base = CacheBase('sqlite')
    """
    return scalar_manager.CacheBase.get(name, **kwargs)
