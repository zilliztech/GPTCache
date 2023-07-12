import os
import random
import time

import numpy

from gptcache import cache, Cache, Config
from gptcache.adapter.adapter import adapt
from gptcache.adapter.api import put, get
from gptcache.manager import get_data_manager, manager_factory
from gptcache.processor.post import first, nop
from gptcache.processor.pre import get_prompt
from gptcache.utils.error import NotInitError
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

    def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
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
        post_process_messages_func=first,
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
    def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
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


def test_cache_temperature():
    if os.path.exists("faiss.index"):
        os.remove("faiss.index")
    if os.path.exists("sqlite.db"):
        os.remove("sqlite.db")
    data_manager = manager_factory(
        "sqlite,faiss", vector_params={"dimension": 3, "top_k": 2}
    )
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=lambda x, **_: numpy.ones((3,)).astype("float32"),
        data_manager=data_manager,
        post_process_messages_func=nop,
    )
    assert cache.data_manager.v._top_k == 2
    prompt = "test"
    answer = "test answer"
    for _ in range(5):
        put(prompt=prompt, data=answer)

    answers = get(prompt=prompt, temperature=2.0)
    assert answers is None

    answers = get(prompt=prompt, temperature=1.5)
    assert answers in [None, [answer] * 5]

    answers = get(prompt=prompt, temperature=0.0, top_k=3)
    assert len(answers) == 3

    answers = get(prompt=prompt, temperature=0.0)
    assert len(answers) == 5

    answers = get(prompt=prompt)
    assert len(answers) == 2


def test_input_summarization():
    cache_obj = Cache()

    def embedding_func(x, **_):
        assert len(x.split()) < 40
        return x

    cache_obj.init(
        pre_func=lambda x, **_: x.get("text"),
        embedding_func=embedding_func,
        data_manager=manager_factory(data_dir=str(random.random())),
        config=Config(input_summary_len=40),
    )
    adapt(
        lambda **_: 0,
        lambda **_: 0,
        lambda **_: 0,
        text="A large language model (LLM) is a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning or semi-supervised learning. LLMs emerged around 2018 and perform well at a wide variety of tasks. This has shifted the focus of natural language processing research away from the previous paradigm of training specialized supervised models for specific tasks.",
        cache_obj=cache_obj,
    )

    adapt(
        lambda **_: 0,
        lambda **_: 0,
        lambda **_: 0,
        text="A large language model (LLM)",
        cache_obj=cache_obj,
    )
