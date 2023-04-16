__all__ = ["ObjectBase"]

from gptcache.utils.lazy_import import LazyImport

object_manager = LazyImport(
    "object_manager", globals(), "gptcache.manager.object_data.manager"
)


def ObjectBase(name: str, **kwargs):
    """Generate specific ObjectStorage with the configuration. For example, setting for
       `ObjectBase` (with `name`) to manage LocalObjectStorage, S3 object storage.

    :param name: the name of the object storage, it is support 'local', 's3'.
    :type name: str
    :param path: the cache root of the LocalObjectStorage.
    :type path: str

    :param bucket: the bucket of s3.
    :type bucket: str
    :param path_prefix: s3 object prefix.
    :type path_prefix: str
    :param access_key: the access_key of s3.
    :type access_key: str
    :param secret_key: the secret_key of s3.
    :type secret_key: str

    :return: ObjectStorage.

    Example:
        .. code-block:: python

            from gptcache.manager import ObjectBase

            obj_storage = ObjectBase('local', path='./')
    """
    return object_manager.ObjectBase.get(name, **kwargs)
