# Example

- [How to set the `embedding` function](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-embedding-function)
- [How to set the `data manager` class](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-data-manager-class)
- [How to set the `similarity evaluation` interface](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-similarity-evaluation-interface)
- [Other cache init params](https://github.com/zilliztech/gpt-cache/tree/main/examples#Other-cache-init-params)
- [Benchmark](https://github.com/zilliztech/gpt-cache/tree/main/examples#Benchmark)

## How to set the `embedding` function

> Please note that not all data managers are compatible with an embedding function.

### [Default embedding function](https://github.com/zilliztech/GPTCache/blob/main/examples/embedding/default.py)

Nothing to do. Only `map data` manager can be configured for use. 

```python
def to_embeddings(data, **kwargs):
    return data
```

### Suitable for embedding methods consisting of a cached storage and vector store

**[ONNX](https://github.com/zilliztech/GPTCache/blob/main/examples/embedding/onnx.py)**

> When creating an Embedding object, the model will be loaded. It is important to remember to pass the dimension to the data manager.

```python
from gptcache.core import cache, Config
from gptcache.cache.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Onnx

onnx = Onnx()
data_manager = get_data_manager("sqlite", "faiss", dimension=onnx.dimension)
cache.init(embedding_func=onnx.to_embeddings,
           data_manager=data_manager,
           similarity_evaluation=SearchDistanceEvaluation(),
           )
cache.set_openai_key()
```

> The usage of the following models is similar to the above.

<details>

<summary> OpenAI </summary>

```python
from gptcache.embedding import OpenAI

openai = OpenAI()
# openai.dimension
# openai.to_embeddings
```

</details>

<details>

<summary> Huggingface </summary>

```python
from gptcache.embedding import Huggingface

huggingface = Huggingface()
# huggingface.dimension
# huggingface.to_embeddings
```

</details>

<details>

<summary> Cohere </summary>

```python
from gptcache.embedding import Cohere

cohere = Cohere()
# cohere.dimension
# cohere.to_embeddings
```

</details>

<details>

<summary> SentenceTransformer </summary>

```python
from gptcache.embedding import SBERT

sbert = SBERT()
# sbert.dimension
# sbert.to_embeddings
```

</details>

<details>

<summary> FastText </summary>

```python
from gptcache.embedding import FastText

fast_text = FastText()
# fast_text.dimension
# fast_text.to_embeddings
```

</details>

### Custom embedding

The function has two parameters: the preprocessed string and parameters reserved for user customization. To acquire these parameters, a similar method to the one above is used: `kwargs.get("embedding_func", {})`.

<details>

<summary> <strong>Example code</strong> </summary>

```python
def to_embeddings(data, **kwargs):
    return data
```

```python
class Cohere:

    def __init__(self, model: str="large", api_key: str=None, **kwargs):
        self.co = cohere.Client(api_key)
        self.model = model

        if model in self.dim_dict():
            self.__dimension = self.dim_dict()[model]
        else:
            self.__dimension = None

    def to_embeddings(self, data):
        if not isinstance(data, list):
            data = [data]
        response = self.co.embed(texts=data, model=self.model)
        embeddings = response.embeddings
        return np.array(embeddings).astype('float32').squeeze(0)

    @property
    def dimension(self):
        if not self.__dimension:
            foo_emb = self.to_embeddings("foo")
            self.__dimension = len(foo_emb)
        return self.__dimension

    @staticmethod
    def dim_dict():
        return {
            "large": 4096,
            "small": 1024
        }
```

</details>

> Note that if you intend to use the model, it should be packaged with a class. The model will be loaded when the object is created to avoid unnecessary loading when not in use. This also ensures that the model is not loaded multiple times during program execution.

## How to set the `data manager` class

**MapDataManager, default**

Store all data in a map data structure, using the question as the key.

```python
from gptcache.manager.factory import get_user_data_manager
from gptcache import cache

data_manager = get_user_data_manager('map')
cache.init(data_manager=data_manager)
cache.set_openai_key()
```

**Cached storage and Vector store**

The user's question and answer data can be stored in a general database such as SQLite or MySQL, while the vector obtained through the question text embedding is stored in a separate vector database.

```python
from gptcache import cache
from gptcache.manager.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np

d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')

data_manager = get_data_manager("sqlite", 'faiss', dimension=d)
cache.init(embedding_func=mock_embeddings,
           data_manager=data_manager,
           similarity_evaluation=SearchDistanceEvaluation(),
           )
cache.set_openai_key()
```

Support general database

- SQLite.
- PostgreSQL.
- MySQL.
- MariaDB.
- SQL Server.
- Oracle.
 
> [Example code](https://github.com/zilliztech/GPTCache/blob/main/examples/data_manager/scalar_store.py)

Support vector database

- Milvus
- Zilliz Cloud
- FAISS
- ChromaDB

> [Example code](https://github.com/zilliztech/GPTCache/blob/main/examples/data_manager/vector_store.py)

**Custom Store**

First, you need to implement two interfaces, namely [`CacheStorage`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/base.py) and [`VectorBase`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/base.py), and then create the corresponding data manager through the `get_user_data_manager` method.

Reference: [CacheStorage sqlalchemy](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/sqlalchemy.py) [VectorBase Faiss](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/faiss.py)

```python
from gptcache import cache
from gptcache.manager.factory import get_user_data_manager

data_manager=get_user_data_manager("scalar_vector", scalar_store=CustomGeneralStore(), vector_store=CustomVectorStore())
cache.init(data_manager=data_manager)
```

## How to set the `similarity evaluation` interface

**[ExactMatchEvaluation, default](https://github.com/zilliztech/GPTCache/blob/main/examples/similarity_evaluation/exact_match.py)**

Exact match between two questions, currently only available for map data manager.

```python
from gptcache import cache
from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation

cache.init(
           similarity_evaluation=ExactMatchEvaluation(),
           )
cache.set_openai_key()
```

<details>

<summary> <strong><a href="https://github.com/zilliztech/GPTCache/blob/main/examples/similarity_evaluation/search_distance.py">SearchDistanceEvaluation</a></strong> </summary>

Using search distance to evaluate sentences pair similarity.

```python
from gptcache import cache
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

cache.init(
           similarity_evaluation=SearchDistanceEvaluation(),
           )
cache.set_openai_key()
```

</details>

<details>

<summary> <strong><a href="https://github.com/zilliztech/GPTCache/blob/main/examples/similarity_evaluation/onnx.py">OnnxModelEvaluation</a></strong> </summary>

Using ONNX model to evaluate sentences pair similarity.

```python
from gptcache import cache
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation

cache.init(
           similarity_evaluation=OnnxModelEvaluation(),
           )
cache.set_openai_key()
```

</details>

<details>

<summary> <strong>NumpyNormEvaluation</strong> </summary>

Using Numpy norm to evaluate sentences pair similarity.

```python
from gptcache import cache
from gptcache.similarity_evaluation.np import NumpyNormEvaluation

cache.init(
           similarity_evaluation=NumpyNormEvaluation(),
           )
cache.set_openai_key()
```

</details>

**Custom similarity evaluation**

To meet the requirements, you will need to implement the `SimilarityEvaluation` interface, which consists of two methods: `evaluation` and `range`.

- evaluation, The function takes three input values, namely user request data, cache data, and user-defined data. The last parameter, by using kwargs.get("evaluation_func", {}), is reserved for users.
- range, The return of the range function needs to return two values, which are the minimum and maximum values.

Reference: [similarity evaluation dir](https://github.com/zilliztech/GPTCache/tree/main/gptcache/similarity_evaluation)

<details>

<summary><h2>Other cache init params</h2></summary>

- **cache_enable_func**: determines whether to use the cache. 

    Among them, `args` and `kwargs` represent the original request parameters. If the function returns True, the cache is enabled.

    You can use this function to ensure that the cache is not enabled when the length of the question is too long, as the likelihood of caching the result is low in such cases.

    ```python
    def cache_all(*args, **kwargs):
        return True
    ```

- **pre_embedding_func**: extracts key information from the request and preprocesses it to ensure that the input information for the encoder module's embedding function is simple and accurate.

    The `data` parameter represents the original request dictionary object, while the `kwargs` parameter is used to obtain user-defined parameters. By using `kwargs.get("pre_embedding_func", {})`, the main purpose is to allow users to pass additional parameters at a certain stage of the process.

    For example, it may extract only the last message in the message array of the OpenAI request body, or the first and last messages in the array.

    ```python
    def last_content(data, **kwargs):
        return data.get("messages")[-1]["content"]
    ```

    ```python
    def all_content(data, **kwargs):
        s = ""
        messages = data.get("messages")
        for i, message in enumerate(messages):
            if i == len(messages) - 1:
                s += message["content"]
            else:
                s += message["content"] + "\n"
        return s
    ```

- **config**: includes cache-related configurations, which currently consist of the following: `log_time_func`, `similarity_threshold`, and `similarity_positive`.

  - log_time_func: The function logging time-consuming operations currently detects `embedding` and `search` functions.
  - similarity_threshold
  - similarity_positive: When set to `True`, a higher value indicates a higher degree of similarity. When set to `False`, a lower value indicates a higher degree of similarity.

- **next_cache**: This points to the next cache object, which is useful for implementing multi-level cache functions.

  ```python
  from gptcache import cache, Cache
  from gptcache.manager.factory import get_data_manager 
  
  bak_cache = Cache()
  bak_data_file = "data_map_bak.txt"
  bak_cache.init(data_manager=get_data_manager("map", data_path=bak_data_file))
  
  cache.init(data_manager=get_data_manager("map"),
             next_cache=bak_cache)
  ```
  
## Request cache parameter customization

- **cache_obj**: customize request cache, use global variable cache by default.

```python
onnx = Onnx()
data_manager = get_si_data_manager("sqlite", "faiss", dimension=onnx.dimension)
one_cache = Cache()
one_cache.init(embedding_func=onnx.to_embeddings,
               data_manager=data_manager,
               evaluation_func=pair_evaluation,
               config=Config(
                   similarity_threshold=1,
                   similarity_positive=False,
                    ),
               )

question = "what do you think about chatgpt"

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question}
    ],
    cache_obj=one_cache
)
```

- **cache_context**: Custom cache parameters can be passed separately for each step in the caching process.

```python
question = "what do you think about chatgpt"

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question}
    ],
    cache_context={
      "pre_embedding_func": {},
      "embedding_func": {},
      "search_func": {},
      "evaluation_func": {},
      "save_func": {},
    }
)
```

- **cache_skip**: skip the cache search, but still store the results returned by the LLM model. These stored results can be used to retry when the cached result is unsatisfactory. Additionally, during the startup phase of the cache system, you can avoid performing a cache search altogether and directly save the data, which can then be used for data accumulation.

```python
question = "what do you think about chatgpt"

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question}
    ],
    cache_skip=True
)
```

</details>

For more details, please refer to: [API Reference](https://gptcache.readthedocs.io/)

## [Benchmark](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py)

The benchmark script about the `Sqlite + Faiss + ONNX`

[Test data source](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/mock_data.json): Randomly scrape some information from the webpage (origin), and then let chatgpt produce corresponding data (similar).

- **threshold**: answer evaluation threshold, A smaller value means higher consistency with the content in the cache, a lower cache hit rate, and a lower cache miss hit; a larger value means higher tolerance, a higher cache hit rate, and at the same time also have higher cache misses.
- **positive**: effective cache hit, which means entering `similar` to search and get the same result as `origin`
- **negative**: cache hit but the result is wrong, which means entering `similar` to search and get the different result as `origin`
- **fail count**: cache miss

data file: [mock_data.json](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/mock_data.json)
similarity evaluation func: pair_evaluation (search distance)

 | threshold | average time | positive | negative | fail count |
|-----------|--------------|----------|----------|------------|
| 20        | 0.04s        | 455      | 27       | 517        |
| 50        | 0.09s        | 871      | 86       | 42         |
| 100       | 0.12s        | 905      | 93       | 1          |
