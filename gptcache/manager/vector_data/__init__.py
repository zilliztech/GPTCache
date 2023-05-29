__all__ = ["VectorBase"]

from gptcache.utils.lazy_import import LazyImport

vector_manager = LazyImport(
    "vector_manager", globals(), "gptcache.manager.vector_data.manager"
)


def VectorBase(name: str, **kwargs):
    """Generate specific VectorBase with the configuration.
    """
    return vector_manager.VectorBase.get(name, **kwargs)
