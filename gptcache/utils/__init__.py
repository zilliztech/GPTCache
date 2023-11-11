__all__ = [
    "import_pymilvus",
    "import_milvus_lite",
    "import_sbert",
    "import_cohere",
    "import_fasttext",
    "import_huggingface",
    "import_uform",
    "import_usearch",
    "import_torch",
    "import_huggingface_hub",
    "import_onnxruntime",
    "import_faiss",
    "import_hnswlib",
    "import_chromadb",
    "import_sqlalchemy",
    "import_sql_client",
    "import_mongodb",
    "import_pydantic",
    "import_langchain",
    "import_pillow",
    "import_boto3",
    "import_diffusers",
    "import_torchaudio",
    "import_torchvision",
    "import_timm",
    "import_vit",
    "import_replicate",
    "import_stability",
    "import_scipy",
    "import_llama_cpp_python",
    "import_ruamel",
    "import_selective_context",
    "import_httpx",
    "import_openai",
    "import_docarray",
    "softmax",
    "import_paddle",
    "import_paddlenlp",
    "import_tiktoken",
    "import_fastapi",
    "import_redis",
    "import_qdrant",
    "import_weaviate",
    ]

import importlib.util
from typing import Optional

from gptcache.utils.dependency_control import prompt_install
from gptcache.utils.softmax import softmax  # pylint: disable=unused-argument


def _check_library(libname: str, prompt: bool = True, package: Optional[str] = None):
    is_avail = False
    if importlib.util.find_spec(libname):
        is_avail = True
    if not is_avail and prompt:
        prompt_install(package if package else libname)
    return is_avail


def import_pymilvus():
    _check_library("pymilvus")


def import_milvus_lite():
    _check_library("milvus")


def import_qdrant():
    _check_library("qdrant_client")


def import_sbert():
    _check_library("sentence_transformers", package="sentence-transformers")


def import_cohere():
    _check_library("cohere")


def import_fasttext():
    _check_library("fasttext", package="fasttext==0.9.2")


def import_huggingface():
    _check_library("transformers")


def import_uform():
    _check_library("uform", package="uform==0.2.1")


def import_usearch():
    _check_library("usearch", package="usearch==0.22.3")


def import_torch():
    _check_library("torch")


def import_huggingface_hub():
    _check_library("huggingface_hub", package="huggingface-hub")


def import_onnxruntime():
    _check_library("onnxruntime", package="onnxruntime==1.14.1")


def import_faiss():
    _check_library("faiss", package="faiss-cpu")


def import_hnswlib():
    _check_library("hnswlib")


def import_chromadb():
    _check_library("chromadb", package="chromadb==0.3.26")


def import_sqlalchemy():
    _check_library("sqlalchemy")


def import_postgresql():
    _check_library("psycopg2", package="psycopg2-binary")


def import_pymysql():
    _check_library("pymysql")


# `brew install unixodbc` in mac
# and install PyODBC driver.
def import_pyodbc():
    _check_library("pyodbc")


# install cx-Oracle driver.
def import_cxoracle():
    _check_library("cx_Oracle")


def import_duckdb():
    _check_library("duckdb", package="duckdb")
    _check_library("duckdb-engine", package="duckdb-engine")


def import_sql_client(db_name):
    if db_name == "postgresql":
        import_postgresql()
    elif db_name in ["mysql", "mariadb"]:
        import_pymysql()
    elif db_name == "sqlserver":
        import_pyodbc()
    elif db_name == "oracle":
        import_cxoracle()
    elif db_name == "duckdb":
        import_duckdb()


def import_mongodb():
    _check_library("pymongo")
    _check_library("mongoengine")


def import_pydantic():
    _check_library("pydantic")


def import_langchain():
    _check_library("langchain")


def import_pillow():
    _check_library("PIL", package="pillow")


def import_boto3():
    _check_library("boto3")


def import_diffusers():
    _check_library("diffusers")


def import_torchaudio():
    _check_library("torchaudio")


def import_torchvision():
    _check_library("torchvision")


def import_timm():
    _check_library("timm", package="timm")


def import_vit():
    _check_library("vit", package="vit")


def import_replicate():
    _check_library("replicate")


def import_stability():
    _check_library("stability_sdk", package="stability-sdk")


def import_scipy():
    _check_library("scipy")


def import_llama_cpp_python():
    _check_library("llama_cpp", package="llama-cpp-python")


def import_ruamel():
    _check_library("ruamel-yaml")


def import_selective_context():
    _check_library("selective_context")


def import_httpx():
    _check_library("httpx")


def import_openai():
    _check_library("openai", package="openai==0.28.1")


def import_docarray():
    _check_library("docarray")


def import_paddle():
    prompt_install("protobuf==3.20.0")
    _check_library("paddlepaddle")


def import_paddlenlp():
    _check_library("paddlenlp")


def import_tiktoken():
    _check_library("tiktoken")


def import_fastapi():
    _check_library("uvicorn", package="'uvicorn[standard]'")
    _check_library("fastapi")


def import_redis():
    _check_library("redis")
    _check_library("redis_om", package="redis-om")


def import_starlette():
    _check_library("starlette")


def import_weaviate():
    _check_library("weaviate-client")
