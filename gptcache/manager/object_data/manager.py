from gptcache.utils.error import NotFoundError


class ObjectBase:
    """
    ObjectBase to manager the object storage.
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
