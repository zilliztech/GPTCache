from gptcache.utils.error import NotFoundError


class ObjectBase:
    """
    ObjectBase to manager the object storage.

    Generate specific ObjectStorage with the configuration. For example, setting for
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

    def __init__(self):
        raise EnvironmentError(
            "CacheBase is designed to be instantiated, please using the `CacheBase.get(name)`."
        )

    @staticmethod
    def get(name, **kwargs):
        if name == "local":
            from gptcache.manager.object_data.local_storage import LocalObjectStorage  # pylint: disable=import-outside-toplevel
            object_base = LocalObjectStorage(kwargs.get("path", "./local_obj"))
        elif name == "s3":
            from gptcache.manager.object_data.s3_storage import S3Storage  # pylint: disable=import-outside-toplevel
            object_base = S3Storage(kwargs.get("path_prefix"), kwargs.get("bucket"),
                                    kwargs.get("access_key"), kwargs.get("secret_key"),
                                    kwargs.get("endpoint"))
        else:
            raise NotFoundError("object store", name)
        return object_base
