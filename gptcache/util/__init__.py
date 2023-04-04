__all__ = ['import_pymilvus', 'import_towhee', 'import_faiss']

from .dependency_control import prompt_install


# pylint: disable=unused-import
# pylint: disable=ungrouped-imports
# pragma: no cover
def import_pymilvus():
    try:
        import pymilvus
    except ModuleNotFoundError as e:
        prompt_install('pymilvus')
        import pymilvus


def import_towhee():
    try:
        import towhee
    except ModuleNotFoundError as e:
        prompt_install('towhee==0.9.0')
        import towhee


def import_faiss():
    try:
        import faiss
    except ModuleNotFoundError as e:
        prompt_install('faiss-cpu==1.6.5')
        import faiss


def import_chromadb():
    try:
        import chromadb
    except ModuleNotFoundError as e:
        prompt_install('chromadb')
        import chromadb


def import_sqlalchemy():
    try:
        import sqlalchemy
    except ModuleNotFoundError as e:
        prompt_install('sqlalchemy')
        import sqlalchemy
