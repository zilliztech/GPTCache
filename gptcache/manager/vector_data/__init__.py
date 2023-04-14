__all__ = ["VectorBase"]

from gptcache.utils.lazy_import import LazyImport

vector_manager = LazyImport(
    "vector_manager", globals(), "gptcache.manager.vector_data.manager"
)


def VectorBase(name: str, **kwargs):
    """Generate specific VectorBase with the configuration. For example, setting for
       `Milvus` (with , `host`, `port`, `password`, `secure`, `collection_name`, `index_params`, `search_params`, `local_mode`, `local_data` params),
       `Faiss` (with , `index_path`, `dimension`, `top_k` params),
       `Chromadb` (with `top_k`, `client_settings`, `persist_directory`, `collection_name` params),
       `Hnswlib` (with `index_file_path`, `dimension`, `top_k`, `max_elements` params).

    :param name: the name of the vectorbase, it is support 'milvus', 'faiss', 'chromadb', 'hnswlib' now.
    :type name: str

    :param top_k: the umber of the vectors results to return, defaults to 1.
    :type top_k: int

    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int

    :param index_path: the path to Faiss index, defaults to 'faiss.index'.
    :type index_path: str

    :param host: the host for Milvus vector database, defaults to 'localhost'.
    :type host: str
    :param port: the port for Milvus vector database, defaults to '19530'.
    :type port: str
    :param user: the user for Zilliz Cloud, defaults to "".
    :type user: str
    :param password: the password for Zilliz Cloud, defaults to "".
    :type password: str
    :param secure: whether it is https with Zilliz Cloud, defaults to False.
    :type secures: bool
    :param index_params: the index parameters for Milvus, defaults to the HNSW index: {'metric_type': 'L2', 'index_type': 'HNSW', 'params': {'M':
                         8, 'efConstruction': 64}}.
    :type index_params: dict
    :param search_params: the index parameters for Milvus, defaults to None.
    :type search_params: dict
    :param collection_name: the name of the collection for Milvus vector database, defaults to 'gptcache'.
    :type collection_name: str
    :param local_mode: if true, will start a local milvus server.
    :type local_mode: bool
    :param local_data: required when local_mode is True.
    :type local_data: str

    :param client_settings: the setting for Chromadb.
    :type client_settings: Settings
    :param persist_directory: the directory to persist, defaults to '.chromadb/' in the current directory.
    :type persist_directory: str

    :param index_path: the path to hnswlib index, defaults to 'hnswlib_index.bin'.
    :type index_path: str
    :param max_elements: max_elements of hnswlib, defaults 100000.
    :type max_elements: int
    """
    return vector_manager.VectorBase.get(name, **kwargs)
