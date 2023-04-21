__all__ = [
    "import_pymilvus",
    "import_milvus_lite",
    "import_sbert",
    "import_cohere",
    "import_fasttext",
    "import_huggingface",
    "import_torch",
    "import_huggingface_hub",
    "import_onnxruntime",
    "import_faiss",
    "import_hnswlib",
    "import_chromadb",
    "import_sqlalchemy",
    "import_sql_client",
    "import_pydantic",
    "import_langchain",
    "import_pillow",
    "import_boto3",
    "import_diffusers",
    "import_torchaudio",
    "import_torchvision",
    "import_timm",
    "import_vit",
    ]

import importlib.util
from typing import Optional

from gptcache.utils.dependency_control import prompt_install


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


def import_sbert():
    _check_library("sentence_transformers", package="sentence-transformers")


def import_cohere():
    _check_library("cohere")


def import_fasttext():
    _check_library("fasttext")


def import_huggingface():
    _check_library("transformers")


def import_torch():
    _check_library("torch")


def import_huggingface_hub():
    _check_library("huggingface_hub", package="huggingface-hub")


def import_onnxruntime():
    _check_library("onnxruntime")


def import_faiss():
    _check_library("faiss", package="faiss-cpu")


def import_hnswlib():
    _check_library("hnswlib")


def import_chromadb():
    _check_library("chromadb")


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


def import_sql_client(db_name):
    if db_name == "postgresql":
        import_postgresql()
    elif db_name in ["mysql", "mariadb"]:
        import_pymysql()
    elif db_name == "sqlserver":
        import_pyodbc()
    elif db_name == "oracle":
        import_cxoracle()


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
