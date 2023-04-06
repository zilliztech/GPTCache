## 😊 Quick Start

**Note**:
- You can quickly try GPT cache and put it into a production environment without heavy development. However, please note that the repository is still under heavy development.
- By default, only a limited number of libraries are installed to support the basic cache functionalities. When you need to use additional features, the related libraries will be **automatically installed**.
- If you encounter issues installing a library due to a low pip version, run: `python -m pip install --upgrade pip`.

### pip install

```bash
pip install gptcache
```

### dev install

```bash
# clone gpt cache repo
git clone https://github.com/zilliztech/gptcache
cd gptcache

# install the repo
pip install -r requirements.txt
python setup.py install
```

### example usage

If you just want to achieve precise matching cache of requests, that is, two identical requests, you **ONLY** need **TWO** steps to access this cache

1. Cache init

```python
from gptcache.core import cache

cache.init()
# If you use the `openai.api_key = xxx` to set the api key, you need use `cache.set_openai_key()` to replace it.
# it will read the `OPENAI_API_KEY` environment variable and set it to ensure the security of the key.
cache.set_openai_key()
```
2. Replace the original openai package

```python
from gptcache.adapter import openai

# openai requests DON'T need ANY changes
answer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ],
)
```

If you want to experience vector similarity search cache locally, you can use the example [Sqlite + Faiss + ONNX](https://github.com/zilliztech/GPTCache/blob/main/examples/sqlite_faiss_onnx/sqlite_faiss_onnx.py).

More Docs：
- [System Design, how it was constructed](./system.md)
- [Features, all features currently supported by the cache](./feature.md)
- [Examples, learn better custom caching](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md)