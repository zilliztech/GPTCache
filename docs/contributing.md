# 😆 Contributing to GPT Cache

Before contributing to GPT Cache, it is recommended to read the [system design article](./system.md).

In the process of contributing, pay attention to **the parameter type**, because there is currently no type restriction added.

First check which part you want to contribute:
- Add a scalar store type
- Add a vector store type
- Add a vector index type
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
# add new method to util/__init__.py
__all__ = ['import_pymilvus']

from .dependency_control import prompt_install


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

## Add a scalar store type

refer to the implementation of [sqlite](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/scalar_data/sqllite3.py).

1. Implement the [ScalarStore](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/scalar_data/scalar_store.py) interface
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
4. Add the new store to the [_get_scalar_store](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/factory.py) method
5. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a vector store type

refer to the implementation of [milvus](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/vector_data/milvus.py).

1. Implement the [VectorStore](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/vector_data/vector_store.py) interface
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Add the new store to the [get_ss_data_manager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/factory.py) method
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a vector index type

refer to the implementation of [faiss](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/vector_data/faiss.py).

1. Implement the [VectorIndex](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/vector_data/vector_index.py) interface
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Add the new store to the [get_si_data_manager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/factory.py) method
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a new data manager

refer to the implementation of [MapDataManager, SSDataManager or SIDataManager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/data_manager.py).

1. Implement the [DataManager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/data_manager.py) interface
2. Add the new store to the [get_data_manager](https://github.com/zilliztech/GPTCache/blob/main/gptcache/cache/factory.py) method
3. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a embedding function

refer to the implementation of [onnx](https://github.com/zilliztech/GPTCache/blob/main/gptcache/embedding/onnx.py) or [openai](https://github.com/zilliztech/GPTCache/blob/main/gptcache/embedding/openai.py).

1. Add a new python file to [embedding](https://github.com/zilliztech/GPTCache/blob/main/gptcache/embedding) directory
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Implement the embedding function and **make sure** your output dimension
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a similarity evaluation function

refer to the implementation of [pair_evaluation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/simple.py) or [onnx](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/onnx.py)

1. Make sure the input params, you can learn more about in the [user view](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/openai.py) model
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
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Implement the similarity evaluation function
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

## Add a method to post-process the cache answer list

refer to the implementation of [first or random_one](https://github.com/zilliztech/GPTCache/blob/main/gptcache/processor/post.py)

1. Make sure the input params, you can learn more about in the [user view](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/openai.py) model
2. Make sure the newly added third-party libraries are lazy imported and automatic installation
3. Implement the post method
4. Add a usage example to [example](https://github.com/zilliztech/GPTCache/blob/main/examples) directory and add the corresponding content to [example.md](https://github.com/zilliztech/GPTCache/blob/main/examples/example.md) [README.md](https://github.com/zilliztech/GPTCache/blob/main/README.md)

# Add a new process in handling chatgpt requests

1. Need to have a clear understanding of the current process, refer to the [user view](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/openai.py) model
2. Add a new process
3. Make sure all examples work properly
