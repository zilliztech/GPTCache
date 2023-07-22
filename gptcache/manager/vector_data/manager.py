from gptcache.utils.error import NotFoundError, ParamError

TOP_K = 1

FAISS_INDEX_PATH = "faiss.index"
DIMENSION = 0

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_USER = ""
MILVUS_PSW = ""
MILVUS_SECURE = False
MILVUS_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}

PGVECTOR_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
PGVECTOR_INDEX_PARAMS = {"index_type": "L2", "params": {"lists": 100, "probes": 10}}

QDRANT_GRPC_PORT = 6334
QDRANT_HTTP_PORT = 6333
QDRANT_INDEX_PARAMS = {"ef_construct": 100, "m": 16}
QDRANT_DEFAULT_LOCATION = "./qdrant_data"
QDRANT_FLUSH_INTERVAL_SEC = 5

COLLECTION_NAME = "gptcache"

WEAVIATE_TIMEOUT_CONFIG = (10, 60)
WEAVIATE_STARTUP_PERIOD = 5


# pylint: disable=import-outside-toplevel
class VectorBase:
    """
    VectorBase to manager the vector base.

    Generate specific VectorBase with the configuration. For example, setting for
       `Milvus` (with , `host`, `port`, `password`, `secure`, `collection_name`, `index_params`, `search_params`, `local_mode`, `local_data` params),
       `Faiss` (with , `index_path`, `dimension`, `top_k` params),
       `Chromadb` (with `top_k`, `client_settings`, `persist_directory`, `collection_name` params),
       `Hnswlib` (with `index_file_path`, `dimension`, `top_k`, `max_elements` params).
       `pgvector` (with `url`, `collection_name`, `index_params`, `top_k`, `dimension` params).

    :param name: the name of the vectorbase, it is support 'milvus', 'faiss', 'chromadb', 'hnswlib' now.
    :type name: str

    :param top_k: the number of the vectors results to return, defaults to 1.
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

    :param url: the connection url for PostgreSQL database, defaults to 'postgresql://postgres@localhost:5432/postgres'
    :type url: str
    :param index_params: the index parameters for pgvector.
    :type index_params: dict
    :param collection_name: the prefix of the table for PostgreSQL pgvector, defaults to 'gptcache'.
    :type collection_name: str

    :param client_settings: the setting for Chromadb.
    :type client_settings: Settings
    :param persist_directory: the directory to persist, defaults to '.chromadb/' in the current directory.
    :type persist_directory: str

    :param index_path: the path to hnswlib index, defaults to 'hnswlib_index.bin'.
    :type index_path: str
    :param max_elements: max_elements of hnswlib, defaults 100000.
    :type max_elements: int
    """

    def __init__(self):
        raise EnvironmentError(
            "VectorBase is designed to be instantiated, please using the `VectorBase.get(name)`."
        )

    @staticmethod
    def check_dimension(dimension):
        if dimension <= 0:
            raise ParamError(
                f"the dimension should be greater than zero, current value: {dimension}."
            )

    @staticmethod
    def get(name, **kwargs):
        top_k = kwargs.get("top_k", TOP_K)
        if name == "milvus":
            from gptcache.manager.vector_data.milvus import Milvus

            dimension = kwargs.get("dimension", DIMENSION)
            VectorBase.check_dimension(dimension)
            host = kwargs.get("host", MILVUS_HOST)
            port = kwargs.get("port", MILVUS_PORT)
            user = kwargs.get("user", MILVUS_USER)
            password = kwargs.get("password", MILVUS_PSW)
            secure = kwargs.get("secure", MILVUS_SECURE)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)
            index_params = kwargs.get("index_params", MILVUS_INDEX_PARAMS)
            search_params = kwargs.get("search_params", None)
            local_mode = kwargs.get("local_mode", False)
            local_data = kwargs.get("local_data", "./milvus_data")
            vector_base = Milvus(
                host=host,
                port=port,
                user=user,
                password=password,
                secure=secure,
                collection_name=collection_name,
                dimension=dimension,
                top_k=top_k,
                index_params=index_params,
                search_params=search_params,
                local_mode=local_mode,
                local_data=local_data,
            )
        elif name == "faiss":
            from gptcache.manager.vector_data.faiss import Faiss

            dimension = kwargs.get("dimension", DIMENSION)
            index_path = kwargs.pop("index_path", FAISS_INDEX_PATH)
            VectorBase.check_dimension(dimension)
            vector_base = Faiss(
                index_file_path=index_path, dimension=dimension, top_k=top_k
            )
        elif name == "chromadb":
            from gptcache.manager.vector_data.chroma import Chromadb

            client_settings = kwargs.get("client_settings", None)
            persist_directory = kwargs.get("persist_directory", None)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)
            vector_base = Chromadb(
                client_settings=client_settings,
                persist_directory=persist_directory,
                collection_name=collection_name,
                top_k=top_k,
            )
        elif name == "hnswlib":
            from gptcache.manager.vector_data.hnswlib_store import Hnswlib

            dimension = kwargs.get("dimension", DIMENSION)
            index_path = kwargs.pop("index_path", "./hnswlib_index.bin")
            max_elements = kwargs.pop("max_elements", 100000)
            VectorBase.check_dimension(dimension)
            vector_base = Hnswlib(
                index_file_path=index_path,
                dimension=dimension,
                top_k=top_k,
                max_elements=max_elements,
            )
        elif name == "pgvector":
            from gptcache.manager.vector_data.pgvector import PGVector

            dimension = kwargs.get("dimension", DIMENSION)
            url = kwargs.get("url", PGVECTOR_URL)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)
            index_params = kwargs.get("index_params", PGVECTOR_INDEX_PARAMS)
            vector_base = PGVector(
                dimension=dimension,
                top_k=top_k,
                url=url,
                collection_name=collection_name,
                index_params=index_params,
            )
        elif name == "docarray":
            from gptcache.manager.vector_data.docarray_index import DocArrayIndex

            index_path = kwargs.pop("index_path", "./docarray_index.bin")
            vector_base = DocArrayIndex(index_file_path=index_path, top_k=top_k)
        elif name == "usearch":
            from gptcache.manager.vector_data.usearch import USearch

            dimension = kwargs.get("dimension", DIMENSION)
            index_path = kwargs.pop("index_path", "./index.usearch")
            metric = kwargs.get("metric", "cos")
            dtype = kwargs.get("dtype", "f32")
            vector_base = USearch(
                index_file_path=index_path,
                dimension=dimension,
                top_k=top_k,
                metric=metric,
                dtype=dtype,
            )
        elif name == "redis":
            from gptcache.manager.vector_data.redis_vectorstore import RedisVectorStore
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", "6379")
            user = kwargs.get("user", "")
            password = kwargs.get("password", "")
            namespace = kwargs.get("namespace", "")
            dimension = kwargs.get("dimension", DIMENSION)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)

            vector_base = RedisVectorStore(
                host=host,
                port=port,
                username=user,
                password=password,
                namespace=namespace,
                dimension=dimension,
                collection_name=collection_name,
                top_k=top_k,
            )
        elif name == "qdrant":
            from gptcache.manager.vector_data.qdrant import QdrantVectorStore
            url = kwargs.get("url", None)
            port = kwargs.get("port", QDRANT_HTTP_PORT)
            grpc_port = kwargs.get("grpc_port", QDRANT_GRPC_PORT)
            prefer_grpc = kwargs.get("prefer_grpc", False)
            https = kwargs.get("https", False)
            api_key = kwargs.get("api_key", None)
            prefix = kwargs.get("prefix", None)
            timeout = kwargs.get("timeout", None)
            host = kwargs.get("host", None)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)
            location = kwargs.get("location", QDRANT_DEFAULT_LOCATION)
            dimension = kwargs.get("dimension", DIMENSION)
            top_k: int = kwargs.get("top_k", TOP_K)
            flush_interval_sec = kwargs.get("flush_interval_sec", QDRANT_FLUSH_INTERVAL_SEC)
            index_params = kwargs.get("index_params", QDRANT_INDEX_PARAMS)
            vector_base = QdrantVectorStore(
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                collection_name=collection_name,
                location=location,
                dimension=dimension,
                top_k=top_k,
                flush_interval_sec=flush_interval_sec,
                index_params=index_params,
            )
        elif name == "weaviate":
            from gptcache.manager.vector_data.weaviate import Weaviate

            url = kwargs.get("url", None)
            auth_client_secret = kwargs.get("auth_client_secret", None)
            timeout_config = kwargs.get("timeout_config", WEAVIATE_TIMEOUT_CONFIG)
            proxies = kwargs.get("proxies", None)
            trust_env = kwargs.get("trust_env", False)
            additional_headers = kwargs.get("additional_headers", None)
            startup_period = kwargs.get("startup_period", WEAVIATE_STARTUP_PERIOD)
            embedded_options = kwargs.get("embedded_options", None)
            additional_config = kwargs.get("additional_config", None)

            vector_base = Weaviate(
                url=url,
                auth_client_secret=auth_client_secret,
                timeout_config=timeout_config,
                proxies=proxies,
                trust_env=trust_env,
                additional_headers=additional_headers,
                startup_period=startup_period,
                embedded_options=embedded_options,
                additional_config=additional_config,
                top_k=top_k,
            )
        else:
            raise NotFoundError("vector store", name)
        return vector_base
