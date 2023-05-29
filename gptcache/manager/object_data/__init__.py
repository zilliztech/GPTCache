__all__ = ["ObjectBase"]

from gptcache.utils.lazy_import import LazyImport

object_manager = LazyImport(
    "object_manager", globals(), "gptcache.manager.object_data.manager"
)


def ObjectBase(name: str, **kwargs):
    """Generate specific ObjectStorage with the configuration. For example, setting for
       `ObjectBase` (with `name`) to manage LocalObjectStorage, S3 object storage.
    """
    return object_manager.ObjectBase.get(name, **kwargs)
