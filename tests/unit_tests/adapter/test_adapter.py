import os
import time

from gptcache.manager import get_data_manager
from gptcache.utils.error import NotInitError
from gptcache.adapter.adapter import adapt
from gptcache import cache, Cache
from gptcache.utils.time import time_cal

data_map_path = "data_map.txt"


def test_adapt():
    def llm_handler(*llm_args, **llm_kwargs):
        a = llm_kwargs.get("a", 0)
        b = llm_kwargs.get("b", 0)
        time.sleep(1)
        return a + b

    def cache_data_convert(cache_data):
        return int(cache_data)

    def update_cache_callback(llm_data, update_cache_func):
        update_cache_func(str(llm_data))
        return llm_data

    def add_llm(*args, **kwargs):
        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )

    def pre_embedding(data, **kwargs):
        a = data.get("a", 0)
        b = data.get("b", 0)
        return f"{a}+{b}"

    if os.path.isfile(data_map_path):
        os.remove(data_map_path)
    map_manager = get_data_manager()
    cache.init(pre_embedding_func=pre_embedding, data_manager=map_manager)

    def report_func(delta_time):
        assert 0.9 < delta_time < 1.1, delta_time

    def add1(**kwargs):
        res = add_llm(a=1, b=2, **kwargs)
        assert res == 3, res

    # pre_embedding -> embedding -> handle
    # 0 + 0 + 1.0
    time_cal(add1, report_func=report_func)()

    # test cache_skip param
    def delay_embedding(data, **kwargs):
        time.sleep(0.5)
        return data

    cache.init(
        pre_embedding_func=pre_embedding,
        embedding_func=delay_embedding,
        data_manager=map_manager,
    )

    def report_func(delta_time):
        assert 1.4 < delta_time < 1.6, delta_time

    # pre_embedding -> embedding -> handle
    # 0 + 0.5 + 1.0
    time_cal(add1, report_func=report_func)(cache_skip=True)

    def report_func(delta_time):
        assert delta_time < 0.6, delta_time

    time_cal(add1, report_func=report_func)()

    time_cal(add1, report_func=report_func)(cache_factor=0)

    time_cal(add1, report_func=report_func)(cache_factor=10)

    # test cache_enable_func
    def update_cache_callback(llm_data, update_cache_func):
        time.sleep(0.5)
        update_cache_func(str(llm_data))
        return llm_data

    def disable_cache(*args, **kwargs):
        return False

    def report_func(delta_time):
        assert 0.9 < delta_time < 1.1, delta_time

    def add_llm(*args, **kwargs):
        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )

    def add2(**kwargs):
        res = add_llm(a=1, b=2, **kwargs)
        assert res == 3, res

    cache.init(
        cache_enable_func=disable_cache,
        pre_embedding_func=pre_embedding,
        embedding_func=delay_embedding,
        data_manager=map_manager,
    )
    time_cal(add2, report_func=report_func)()


def test_not_init_cache():
    foo_cache = Cache()
    is_exception = False
    try:
        adapt(None, None, None, cache_obj=foo_cache)
    except NotInitError:
        is_exception = True

    assert is_exception
