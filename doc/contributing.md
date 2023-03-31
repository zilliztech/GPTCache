# Contributing to GPT Cache

Before contributing to GPT Cache, it is recommended to read the [system design article](system.md). 

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

## Add a scalar store type

refer to the implementation of [sqlite](../gpt_cache/cache/scalar_data/sqllite3.py).

1. Implement the [ScalarStore](../gpt_cache/cache/scalar_data/scalar_store.py) interface
2. Make sure the newly added third-party libraries are lazy loaded, change the [scalar_data/__init__.py](../gpt_cache/cache/scalar_data/__init__.py), refer to: [vector_data/__init__.py](../gpt_cache/cache/vector_data/__init__.py)
3. Add the new store to the [_get_scalar_store](../gpt_cache/cache/factory.py) method
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a vector store type

refer to the implementation of [milvus](../gpt_cache/cache/vector_data/milvus.py).

1. Implement the [VectorStore](../gpt_cache/cache/vector_data/vector_store.py) interface
2. Make sure the newly added third-party libraries are lazy loaded, change the [vector_data/__init__.py](../gpt_cache/cache/vector_data/__init__.py)
3. Add the new store to the [get_ss_data_manager](../gpt_cache/cache/factory.py) method
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a vector index type

refer to the implementation of [faiss](../gpt_cache/cache/vector_data/faiss.py).

1. Implement the [VectorIndex](../gpt_cache/cache/vector_data/vector_index.py) interface
2. Make sure the newly added third-party libraries are lazy loaded, change the [vector_data/__init__.py](../gpt_cache/cache/vector_data/__init__.py)
3. Add the new store to the [get_si_data_manager](../gpt_cache/cache/factory.py) method
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a new data manager

refer to the implementation of [MapDataManager, SSDataManager or SIDataManager](../gpt_cache/cache/data_manager.py).

1. Implement the [DataManager](../gpt_cache/cache/data_manager.py) interface
2. Add the new store to the [get_data_manager](../gpt_cache/cache/factory.py) method
3. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
4. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a embedding function

refer to the implementation of [towhee](../gpt_cache/embedding/towhee.py) or [openai](../gpt_cache/embedding/openai.py).

1. Add a new python file to [embedding](../gpt_cache/embedding) directory
2. Make sure the newly added third-party libraries are lazy loaded, change the [embedding/__init__.py](../gpt_cache/embedding/__init__.py)
3. Implement the embedding function and **make sure** your output dimension
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a similarity evaluation function

refer to the implementation of [pair_evaluation](../gpt_cache/similarity_evaluation/simple.py) or [towhee](../gpt_cache/similarity_evaluation/towhee.py)

1. Make sure the input params, you can learn more about in the [user view](../gpt_cache/view/openai.py) model
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
2. Make sure the newly added third-party libraries are lazy loaded, change the [similarity_evaluation/__init__.py](../gpt_cache/similarity_evaluation/__init__.py)
3. Implement the similarity evaluation function
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

## Add a method to post-process the cache answer list

refer to the implementation of [first or random_one](../gpt_cache/post_process/post_process.py)

1. Make sure the input params, you can learn more about in the [user view](../gpt_cache/view/openai.py) model
2. Make sure the newly added third-party libraries are lazy loaded, change the [post_process/__init__.py](../gpt_cache/post_process/__init__.py)
3. Implement the post method
4. Add a usage example to [example](../example) directory and add the corresponding content to [example.md](../example/example.md) [README.md](../README.md)
5. Add the installation method to [Install Dependencies List](installation.md) if a third-party library is newly added

# Add a new process in handling chatgpt requests

1. Need to have a clear understanding of the current process, refer to the [user view](../gpt_cache/view/openai.py) model
2. Add a new process
3. Make sure all examples work properly
