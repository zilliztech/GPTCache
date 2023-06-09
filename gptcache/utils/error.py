class CacheError(Exception):
    """GPTCache base error"""


class NotInitError(CacheError):
    """Raise when the cache has been used before it's inited"""
    def __init__(self):
        super().__init__("The cache should be inited before using")


class NotFoundError(CacheError):
    """Raise when getting an unsupported store."""
    def __init__(self, store_type, current_type_name):
        super().__init__(f"Unsupported ${store_type}: {current_type_name}")


class ParamError(CacheError):
    """Raise when receiving an invalid param."""


class PipInstallError(CacheError):
    """Raise when failed to install package."""
    def __init__(self, package):
        super().__init__(f"Ran into error installing {package}.")


def wrap_error(e: Exception) -> Exception:
    """Add a type to exception `e` while ensuring that the original type is not changed

    Example:
        .. code-block:: python

            import openai

            from gptcache.utils.error import wrap_error


            def raise_error():
                try:
                    raise openai.error.OpenAIError(message="test")
                except openai.error.OpenAIError as e:
                    raise wrap_error(e)


            try:
                raise_error()
            except openai.error.OpenAIError as e:
                print("exception:")
                print(e)

            print("over")
    """
    e.__class__ = type(e.__class__.__name__, (CacheError, e.__class__), {})
    return e
