# 😍 Contributing to GPTCache

Before contributing to GPTCache, it is recommended to read the [usage doc](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md) [example-doc](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md).
These two articles will introduce how to use GPTCache and the meaning of parameters of related functions.

In the process of contributing, pay attention to **the parameter type**, because there is currently no type restriction added.

> Note that development **MUST** be based on the **`dev`** branch

First check which part you want to contribute:
- Add a method to pre-process the llm request
- Add a scalar store type
- Add a vector store type
- Add a new data manager
- Add a embedding function
- Add a similarity evaluation function
- Add a method to post-process the cache answer list
- Add a new process in handling chatgpt requests

## Lazy import and automatic installation

**For newly added third-party dependencies, lazy import and automatic installation are required.** Implementation consists of the following steps:
1. Lazy import
```python
# The __init__.py file of the same directory under the new file
__all__ = ['Milvus']

from gptcache.utils.lazy_import import LazyImport

milvus = LazyImport('milvus', globals(), 'gptcache.cache.vector_data.milvus')


def Milvus(**kwargs):
    return milvus.Milvus(**kwargs)
```
2. Automatic installation
```python
# 2.1 Add the import method
# add new method to utils/__init__.py
__all__ = ['import_pymilvus']

from gptcache.utils.dependency_control import prompt_install


def import_pymilvus():
    try:
        # pylint: disable=unused-import
        import pymilvus
    except ModuleNotFoundError as e:  # pragma: no cover
        prompt_install('pymilvus')
        import pymilvus  # pylint: disable=ungrouped-imports

# 2.2 use the import method in your file
from gptcache.util import import_pymilvus
import_pymilvus()
```

## Add a method to pre-process the llm request

refer to the implementation of [Pre](https://github.com/zilliztech/GPTCache/blob/main/gptcache/processor/pre.py).

1. Make sure the input params, the `data` represents the original request dictionary object
2. Implement the post method
3. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

```python
# The origin openai request
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

# This is the pre-process function of openai request, which is to get the last message
def last_content(data, **_):
    return data.get("messages")[-1]["content"]
```

## Add a cache storage type

refer to the implementation of [SQLDataBase](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/sqlalchemy.py).

1. Implement the [CacheStorage](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/base.py) interface
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
4. Add the new store to the [CacheBase](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/manager.py) method
5. Add a usage example to [example](https://github.com/zilliztech/GPTCache/tree/main/examples/data_manager) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a vector store type

refer to the implementation of [milvus](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/milvus.py).

1. Implement the [VectorBase](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/base.py) interface
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Add the new store to the [VectorBase](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/manager.py) method
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/tree/main/examples/data_manager) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a new data manager

refer to the implementation of [MapDataManager, SSDataManager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/data_manager.py).

1. Implement the [DataManager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/data_manager.py) interface
2. Add the new store to the [get_data_manager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/data_manager.py) method
3. Add a usage example to [example](https://github.com/zilliztech/GPTCache/tree/main/examples/data_manager) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a embedding function

refer to the implementation of [cohere](https://github.com/zilliztech/GPTCache/blob/main/gptcache/embedding/cohere.py) or [openai](https://github.com/zilliztech/GPTCache/blob/main/gptcache/embedding/openai.py).

1. Add a new python file to [embedding](https://github.com/zilliztech/GPTCache/tree/main/gptcache/embedding) directory
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Implement the embedding function and **make sure** your output dimension
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/tree/main/examples/embedding) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a similarity evaluation function

refer to the implementation of [SearchDistanceEvaluation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/distance.py) or [OnnxModelEvaluation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/onnx.py)

1. Implement the [SimilarityEvaluation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/similarity_evaluation.py) interface
2. Make sure the range of return value, the `range` method return the min and max value
3. Make sure the input params of `evaluation`, you can learn more about in the [user view](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/openai.py) model
```python
rank = chat_cache.evaluation_func({
    "question": pre_embedding_data,
    "embedding": embedding_data,
}, {
    "question": cache_question,
    "answer": cache_answer,
    "search_result": cache_data,
}, extra_param=context.get('evaluation', None))
```
4. Make sure the newly added third-party libraries are lazy imported and automatic installation
5. Implement the similarity evaluation function
6. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a method to post-process the cache answer list

refer to the implementation of [first or random_one](https://github.com/zilliztech/GPTCache/blob/main/gptcache/processor/post.py)

1. Make sure the input params, you can learn more about in the [adapter](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/adapter.py)
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Implement the post method
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/README.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

```python
# Get the most similar one from multiple results
def first(messages):
    return messages[0]


# Randomly fetch one of many results
def random_one(messages):
    return random.choice(messages)
```

# Add a new process in handling chatgpt requests

1. Need to have a clear understanding of the current process, refer to the [adapter](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/adapter.py)
2. Add a new process
3. Make sure all examples work properly
