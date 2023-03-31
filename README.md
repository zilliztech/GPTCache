# GPT Cache

English | [中文](README-CN.md)

The GPT Cache system is mainly used to cache the question-answer data of users in ChatGPT. This system brings two benefits:

1. Quick response to user requests: compared to large model inference, searching for data in the caching system will have lower latency, enabling faster response to user requests.
2. Reduced service costs: currently, most ChatGPT services are charged based on the number of requests. If user requests hit the cache, it can reduce the number of requests and thus lower service costs.

If the idea 💡 is helpful to you, please feel free to give me a star 🌟, which is helpful to me.

## 😊 Quickly Start

**Note**:
- You can quickly experience the cache, it is worth noting that maybe this is not very **stable**.
- By default, basically **a few** libraries are installed. When you need to use some features, it will **auto install** related libraries.
- If failed to install a library for low pip version, run: `python -m pip install --upgrade pip` 

### pip install

```bash
pip install gpt_cache
```

### dev install

```bash
# clone gpt cache repo
git clone https://github.com/zilliztech/gpt-cache
cd gpt-cache

# install the repo
pip install -r requirements.txt
python setup.py install
```

### quick usage

If you just want to achieve precise matching cache of requests, that is, two identical requests, you **ONLY** need **TWO** steps to access this cache

1. Cache init

```python
from gpt_cache.core import cache

cache.init()
# If you use the `openai.api_key = xxx` to set the api key, you need use `cache.set_openai_key()` to replace it
cache.set_openai_key()
```
2. Replace the original openai package

```python
from gpt_cache.view import openai

# openai requests DON'T need ANY changes
answer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ],
)
```

If you want to experience vector similarity search cache locally, you can use the example [Sqlite + Faiss + Towhee](example/sqlite_faiss_towhee/sqlite_faiss_towhee.py).

More Docs：
- [System Design, how it was constructed](doc/system.md)
- [Features, all features currently supported by the cache](doc/feature.md)
- [Examples, learn better custom caching](example/example.md)

## 🤔 Is Cache necessary?

I believe it is necessary for the following reasons:

- Many question-answer pairs in certain domain services based on ChatGPT have a certain similarity.
- For a user, there is a certain regularity in the series of questions raised using ChatGPT, which is related to their occupation, lifestyle, personality, etc. For example, the likelihood of a programmer using ChatGPT services is largely related to their work.
- If your ChatGPT service targets a large user group, categorizing them can increase the probability of relevant questions being cached, thus reducing service costs.

## 🤗 All Model

![GPTCache Struct](doc/GPTCacheStructure.png)

- Pre-embedding, get the key information in the request
  - get the last message in the request, see: `pre_embedding.py#last_content`
- Embedding, transfer the text to vector for similarity search
  - [x] [towhee](https://towhee.io/), english model: paraphrase-albert-small-v2, chinese model: uer/albert-base-chinese-cluecorpussmall
  - [x] openai embedding api
  - [x] string, nothing change
  - [ ] [cohere](https://docs.cohere.ai/reference/embed) embedding api  
- Cache, data manager, including search, save or evict
  - scalar store
    - [x] [sqlite](https://sqlite.org/docs.html)
    - [ ] [postgresql](https://www.postgresql.org/)
    - [ ] [mysql](https://www.mysql.com/)
  - vector store
    - [x] [milvus](https://milvus.io/)
    - [x] [zilliz cloud](https://cloud.zilliz.com/)
  - vector index
    - [x] [faiss](https://faiss.ai/)
- Similarity Evaluation, judging the quality of cached answers
  - the search distance, see: `simple.py#pair_evaluation`
  - [towhee](https://towhee.io/), roberta_duplicate, precise comparison of problems to problems mode, only support the 512 token
  - string, the cache request and the original request are judged by the exact match of characters
  - np, use the `linalg.norm`
- Post Process, how multiple cached answers are returned to the user
  - choose the most similar
  - choose randomly


## 😆 Contributing

Want to help build GPT Cache? Check out our [contributing documentation](doc/contributing.md).


## 🙏 Thank

Thanks to my colleagues in the company [Zilliz](https://zilliz.com/) for their inspiration and technical support.