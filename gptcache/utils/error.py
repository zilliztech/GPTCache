class CacheError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NotInitError(CacheError):
    def __init__(self):
        super().__init__("the cache should be inited before using")


class NotFoundStoreError(CacheError):
    def __init__(self, store_type, current_type_name):
        super().__init__(f"Unsupported ${store_type}: {current_type_name}")


class ParamError(CacheError):
    def __init__(self, message):
        super().__init__(message)
