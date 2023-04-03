# 🥳 Feature

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
  - Adapter: The user interface to adapt different LLM model requests to the GPT cache protocol
  - Pre-processor: Extracts the key information from the request and preprocess
  - Context Buffer: Maintains session context
  - Encoder: Embed the text into a dense vector for similarity search
  - Cache manager: which includes searching, saving, or evicting data
  - Ranker: Evaluate similarity by judging the quality of cached answers
  - Post-processor: Determine which cached answers to the user, and generate the response