__all__ = ['import_pymilvus', 'import_towhee', 'import_faiss']

from .dependency_control import prompt_install


def import_pymilvus():
    try:
        # pylint: disable=unused-import
        import pymilvus
    except ModuleNotFoundError as e:  # pragma: no cover
        prompt_install('pymilvus')
        import pymilvus  # pylint: disable=ungrouped-imports


def import_towhee():
    try:
        # pylint: disable=unused-import
        import towhee
    except ModuleNotFoundError as e:  # pragma: no cover
        prompt_install('towhee==0.9.0')
        import towhee  # pylint: disable=ungrouped-imports


def import_faiss():
    try:
        # pylint: disable=unused-import
        import faiss
    except ModuleNotFoundError as e:  # pragma: no cover
        prompt_install('faiss-cpu==1.6.5')
        import faiss  # pylint: disable=ungrouped-imports
