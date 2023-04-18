# 🎉 Introduction to new functions of GPTCache

To read the following content, you need to understand the basic use of GPTCache, references:

- [Readme doc](https://github.com/zilliztech/GPTCache)
- [Usage doc](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md)

## v0.1.15 (2023.4.17)

1. Add GPTCache api, makes it easier to access other different llm models and applications

```python
from gptcache.adapter.api import put, get
from gptcache.processor.pre import get_prompt
from gptcache import cache

cache.init(pre_embedding_func=get_prompt)
put("hello", "foo")
print(get("hello"))
```

2. Add image generation bootcamp, link: https://github.com/zilliztech/GPTCache/blob/main/docs/bootcamp/openai/image_generation.ipynb

## v0.1.14 (2023.4.17)

1. Fix to fail to save the data to cache

## ~~v0.1.13 (2023.4.16)~~ Don't Use it, should use the `v0.1.14`

1. Add openai audio adapter (**experimental**)

```python
cache.init(pre_embedding_func=get_file_bytes)

openai.Audio.transcribe(
    model="whisper-1",
    file=audio_file
)
```

2. Improve data eviction implementation

In the future, users will have greater flexibility to customize eviction methods, such as by using Redis or Memcached. Currently, the default caching library is cachetools, which provides an in-memory cache. Other libraries are not currently supported, but may be added in the future.

## v0.1.12 (2023.4.15)

1. The llm request can customize topk search parameters

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question},
    ],
    top_k=10,
)
```

## v0.1.11 (2023.4.14)

1. Add openai complete adapter

```python
cache.init(pre_embedding_func=get_prompt)

response = openai.Completion.create(
                model="text-davinci-003",
                prompt=question
            )
```

2. Add langchain and openai [bootcamp](https://github.com/zilliztech/GPTCache/tree/main/docs/bootcamp)

3. Add openai image adapter (**experimental**)

```python
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()

prompt1 = 'a cat sitting besides a dog'
size1 = '256x256'

openai.Image.create(
    prompt=prompt1,
    size=size1,
    response_format='b64_json'
    )
```

4. Refine storage interface

## v0.1.10 (2023.4.13)

1. Add kreciprocal similarity evaluation

K-reprciprocl evaluation is a method inspired by the popular reranking method in ReID(https://arxiv.org/abs/1701.08398). The term “k-reciprocal” comes from the fact that the algorithm creates reciprocal relationships between similar embeddings in the top-k list. In other words, if embedding A is similar to embedding B and embedding B is similar to embedding A, then A and B are said to be “reciprocally similar” to each other. This evaluation abandon those embeddings pairs which are not “reciprocally similar” in their K nearest neighbors. And the remaining pairs will keep the distance for the final rank.

```python
vector_base = VectorBase("faiss", dimension=d)
data_manager = get_data_manager(CacheBase("sqlite"), vector_base)
evaluation = KReciprocalEvaluation(vectordb=vector_base)
cache.init(
    ... # other configs
    data_manager=data_manager,
    similarity_evaluation=evaluation,
)
```

2. Add LangChainChat adapter

```python
from gptcache.adapter.langchain_models import LangChainChat

cache.init(
    pre_embedding_func=get_msg,
)

chat = LangChainChat(chat=ChatOpenAI(temperature=0))
answer = chat(
    messages=[
        HumanMessage(
            content="Translate this sentence from English to Chinese. I love programming."
        )
    ]
)
```

## v0.1.9 (2023.4.12)

1. Import data into cache

```python
cache.init()

questions = ["foo1", "foo2"]
answers = ["a1", "a2"]
cache.import_data(questions=questions, answers=answers)
```

2. New pre-process function: remove prompts

When using the LLM model, a prompt may be added for each input. If the entire message with the prompt is brought into the cache, it may lead to an increase in the cache error hit rate. For example, the text of the prompt is very long, and the text of the real question is very short. .

```python
cache_obj.init(
    pre_embedding_func=last_content_without_prompt,
    config=Config(prompts=["foo"]),
)
```

3. Embeded milvus

The embedded Milvus is a lightweight version of Milvus that can be embedded into your Python application. It is a single binary that can be easily installed and run on your machine.

```python
with TemporaryDirectory(dir="./") as root:
    db = VectorBase(
                    "milvus",
                    local_mode=True,
                    local_data=str(root),
                    ... #other config
                )
    data_manager = get_data_manager("sqlite", vector_base)
```