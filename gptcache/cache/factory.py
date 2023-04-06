from .data_manager import DataManager, SSDataManager
from .scalar_data import SQLDataBase, SQL_URL
from .vector_data import Milvus, Faiss, Chromadb
from ..utils.error import NotFoundStoreError, ParamError
from ..utils import import_sql_client


def get_user_data_manager(data_manager_name: str, **kwargs) -> DataManager:
    if data_manager_name == "map":
        from .data_manager import MapDataManager

        return MapDataManager(kwargs.pop("data_path", "data_map.txt"),
                              kwargs.pop("max_size", 100),
                              kwargs.pop("get_data_container", None))
    elif data_manager_name == "scalar_vector":
        scalar_store = kwargs.pop("scalar_store", None)
        vector_store = kwargs.pop("vector_store", None)
        max_size = kwargs.pop("max_size", 1000)
        clean_size = kwargs.pop("clean_size", int(max_size * 0.2))
        if scalar_store is None or vector_store is None:
            raise ParamError(f"Missing scalar_store or vector_store parameter for scalar_vector")
        return SSDataManager(max_size, clean_size, scalar_store, vector_store)
    else:
        raise NotFoundStoreError("data manager", data_manager_name)


def _get_scalar_store(scalar_store: str, **kwargs):
    if scalar_store in ["sqlite", "postgresql", "mysql", "mariadb", "sqlserver", "oracle"]:
        sql_url = kwargs.pop("sql_url", SQL_URL[scalar_store])
        table_name = kwargs.pop("table_name", "gptcache")
        import_sql_client(scalar_store)
        store = SQLDataBase(db_type=scalar_store, url=sql_url, table_name=table_name)
    else:
        raise NotFoundStoreError("scalar store", scalar_store)
    return store


def _get_common_params(**kwargs):
    max_size = kwargs.pop("max_size", 1000)
    clean_size = kwargs.pop("clean_size", int(max_size * 0.2))
    top_k = kwargs.pop("top_k", 1)
    dimension = kwargs.pop("dimension", 0)
    return max_size, clean_size, dimension, top_k


def _check_dimension(dimension):
    if dimension <= 0:
        raise ParamError(f"the data manager should set the 'dimension' parameter greater than zero, "
                         f"current: {dimension}")


# scalar_store + vector_store
def get_data_manager(cache_store: str, vector_store: str, **kwargs):
    """Generate SSDataManager with the cache and vector configuration.

    :param cache_store: the name of the cache storage, it is support "sqlite", "postgresql", "mysql", "mariadb", "sqlserver" and  "oracle" now.
    :type cache_store: str.
    :param vector_store: the name of the vector storage, it is support "milvus", "faiss" and "chromadb" now.
    :type vector_store: str.
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int.
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int.
    :param clean_size: the size to clean up, defaults to `max_size * 0.2`.
    :type clean_size: int.
    :param top_k: the umber of the vectors results to return, defaults to 1.
    :type top_k: int.
    :param sql_url: the url of the sql database for cache, such as "<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>",
                    and the default value is related to the `cache_store` parameter, "sqlite:///./sqlite.db" for "sqlite",
                    "postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres" for "postgresql",
                    "mysql+pymysql://root:123456@127.0.0.1:3306/mysql" for "mysql",
                    "mariadb+pymysql://root:123456@127.0.0.1:3307/mysql" for "mariadb",
                    "mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server" for "sqlserver",
                    "oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8" for "oracle".
    :type sql_url: str.
    :param table_name: the table name for sql database, defaults to "gptcache".
    :type table_name: str.
    :param index_path: the path to Faiss index, defaults to "faiss.index".
    :type index_path: str.
    :param collection_name: the name of the collection for Milvus vector database, defaults to "gptcache".
    :type collection_name: str.
    :param host: the host for Milvus vector database, defaults to "localhost".
    :type host: str.
    :param port: the port for Milvus vector database, defaults to "19530".
    :type port: str.
    :param user: the user for Zilliz Cloud, defaults to "".
    :type user: str.
    :param password: the password for Zilliz Cloud, defaults to "".
    :type password: str.
    :param is_https: whether it is https with Zilliz Cloud, defaults to False.
    :type is_https: bool.
    :param eviction: The eviction policy, it is support "LRU" and "FIFO" now, and defaults to "LRU".
    :type eviction:  str.


    :return: SSDataManager.

    Example:
        .. code-block:: python

            from gptcache.cache.factory import get_ss_data_manager

            data_manager = get_ss_data_manager("sqlite", "faiss", dimension=128)
    """
    max_size, clean_size, dimension, top_k = _get_common_params(**kwargs)
    eviction = kwargs.pop("eviction", "LRU")
    scalar = _get_scalar_store(cache_store, **kwargs)
    if vector_store == "milvus":
        _check_dimension(dimension)
        vector = Milvus(dim=dimension, top_k=top_k, **kwargs)
    elif vector_store == "faiss":
        _check_dimension(dimension)
        index_path = kwargs.pop("index_path", "faiss.index")
        vector = Faiss(index_path, dimension, top_k)
    elif vector_store == "chromadb":
        vector = Chromadb(top_k=top_k, **kwargs)
    else:
        raise NotFoundStoreError("vector store", vector_store)
    return SSDataManager(max_size, clean_size, scalar, vector, eviction)
