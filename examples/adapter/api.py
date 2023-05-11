from gptcache import cache, Config, Cache
from gptcache.adapter.api import put, get, init_similar_cache
from gptcache.processor.post import nop
from gptcache.processor.pre import get_prompt


def run_basic():
    cache.init(pre_embedding_func=get_prompt)
    put("hello", "foo")
    print(get("hello"))
    # output: foo


def run_similar_match():
    inner_cache = Cache()
    init_similar_cache(
        cache_obj=inner_cache, post_func=nop, config=Config(similarity_threshold=0)
    )

    put("hello1", "foo1", cache_obj=inner_cache)
    put("hello2", "foo2", cache_obj=inner_cache)
    put("hello3", "foo3", cache_obj=inner_cache)

    messages = get("hello", cache_obj=inner_cache, top_k=3)
    print(messages)
    # output: ['foo1', 'foo2', 'foo3']


if __name__ == "__main__":
    run_basic()
    run_similar_match()
