class CacheError(Exception):
    """GPTCache base error"""


class NotInitError(CacheError):
    """Raise when the cache has been used before it's inited"""
    def __init__(self):
        super().__init__("the cache should be inited before using")


class NotFoundStoreError(CacheError):
    """Raise when getting an unsupported store."""
    def __init__(self, store_type, current_type_name):
        super().__init__(f"Unsupported ${store_type}: {current_type_name}")


class ParamError(CacheError):
    """Raise when receiving an invalid param."""
