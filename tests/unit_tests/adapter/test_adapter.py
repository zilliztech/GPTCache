import os
import time

from gptcache.adapter.adapter import adapt
from gptcache.core import cache, time_cal

data_map_path = "data_map.txt"


def test_adapt():
    def llm_handler(*llm_args, **llm_kwargs):
        a = llm_kwargs.get("a", 0)
        b = llm_kwargs.get("b", 0)
        time.sleep(1)
        return a + b

    def pre_embedding(data, **kwargs):
        a = data.get("a", 0)
        b = data.get("b", 0)
        return f"{a}+{b}"

    def cache_data_convert(cache_data):
        return int(cache_data)

    def update_cache_callback(llm_data, update_cache_func):
        update_cache_func(str(llm_data))
        return llm_data

    def add_llm(*args, **kwargs):
        return adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs)

    if os.path.isfile(data_map_path):
        os.remove(data_map_path)

    cache.init(pre_embedding_func=pre_embedding)

    def report_func(delta_time):
        assert delta_time > 0.9

    def add1():
        res = add_llm(a=1, b=2)
        assert res == 3, res

    time_cal(add1, report_func=report_func)()

    def report_func(delta_time):
        assert delta_time < 0.2

    time_cal(add1, report_func=report_func)()
