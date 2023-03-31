# 🥳 Feature

English | [中文](feature_cn.md)

- Support the openai chat completion normal and stream request
- Get top_k similar search results, it can be set when creating the data manager
- Support the cache chain, see: `Cache#next_cache`

```python
bak_cache = Cache()
bak_cache.init()
cache.init(next_cache=bak_cache)
```

- Whether to completely skip the current cache, that is, do not search the cache or save the Chat GPT results, see: `Cache#cache_enable_func`
- In the cache initialization phase, no cache search is performed, but save the result returned by the chat gpt to cache, see: `cache_skip=True` in `create` request

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=mock_messages,
    cache_skip=True,
)
```

- Like Lego bricks, custom assemble all modules, including:
  - pre-embedding function, get feature information in the original request, such as prompt, last message, etc.
  - embedding function, convert feature information into a vector for cache search, choose a model that fits your use case
  - data manager, cache data management, mainly dealing with the search and storage of cache data
  - cache similarity evaluation function, can use the distance of similar search or additional selection model to ensure that the answer is more accurate
  - post-process the cache answer list, first, random or custom combination