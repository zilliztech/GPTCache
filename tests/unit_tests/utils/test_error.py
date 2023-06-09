from gptcache.utils.error import (
    CacheError,
    NotInitError,
    NotFoundError,
    ParamError,
)


def test_error_type():
    not_init_error = NotInitError()
    assert issubclass(type(not_init_error), CacheError)

    not_found_store_error = NotFoundError("unittest", "test_error_type")
    assert issubclass(type(not_found_store_error), CacheError)

    param_error = ParamError("unittest")
    assert issubclass(type(param_error), CacheError)


def test_wrap():
    import openai

    from gptcache.utils.error import wrap_error

    def raise_error():
        try:
            raise openai.error.OpenAIError(message="test")
        except openai.error.OpenAIError as e:
            raise wrap_error(e)

    is_exception = False
    try:
        raise_error()
    except openai.error.OpenAIError as e:
        is_exception = True

    assert is_exception
