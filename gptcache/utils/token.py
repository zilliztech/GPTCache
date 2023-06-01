from gptcache.utils import import_tiktoken

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        import_tiktoken()
        import tiktoken  # pylint: disable=C0415
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def token_counter(text):
    """Token Counter"""
    num_tokens = len(_get_encoding().encode(text))
    return num_tokens
