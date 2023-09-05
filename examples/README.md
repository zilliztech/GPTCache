# Example

- [How to run Visual Question Answering with MiniGPT-4](#How-to-run-Visual-Question-Answering-with-MiniGPT-4)
- [How to set the **embedding** function](#How-to-set-the-embedding-function)
- [How to set the **data manager** class](#How-to-set-the-data-manager-class)
- [How to set the **similarity evaluation** interface](#How-to-set-the-similarity-evaluation-interface)
- [Other cache init params](#Other-cache-init-params)
- [How to run with session](#How-to-run-with-session)
- [How to use GPTCache server](#How-to-use-GPTCache-server)
- [Benchmark](#Benchmark)

## How to run Visual Question Answering with MiniGPT-4

You can run [vqa_demo.py](./vqa_demo.py) to implement the image Q&A, which uses MiniGPT-4 for generating answers and then GPTCache to cache the answers.

>  Note that you need to make sure that [minigpt4](https://github.com/Vision-CAIR/MiniGPT-4) and [gptcache](https://gptcache.readthedocs.io/en/dev/index.html) are successfully installed, and move the **vqa_demo.py** file to the MiniGPT-4 directory.

```bash
$ python vqa_demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

The above command will use the exact match cache, i.e. map cache management method. When you ask the same image and question, it will hit the cache directly and return the answer quickly.

If you want to use similar search cache, you can run the following command to set `map` to `False`, which will use sqlite3 and faiss to manage the cache to search for similar images and questions in the cache.

> You can also set `dir` to your workspace directory.

```bash
$ python vqa_demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --dir /path/to/workspace --no-map
```


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
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Onnx

onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
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

<details>

<summary> PaddleNLP </summary>

```python
from gptcache.embedding import PaddleNLP

paddlenlp = PaddleNLP()
# paddlenlp.dimension
# paddlenlp.to_embeddings
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
from gptcache.manager import get_data_manager
from gptcache import cache

data_manager = get_data_manager()
cache.init(data_manager=data_manager)
cache.set_openai_key()
```

**Cached storage and Vector store**

The user's question and answer data can be stored in a general database such as SQLite or MySQL, while the vector obtained through the question text embedding is stored in a separate vector database.

```python
from gptcache import cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np

d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')

cache_base = CacheBase('sqlite')
vector_base = VectorBase('faiss', dimension=d)
data_manager = get_data_manager(cache_base, vector_base)
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
- DynamoDB

> [Example code](https://github.com/zilliztech/GPTCache/blob/main/examples/data_manager/scalar_store.py)

Support vector database

- Milvus
- Zilliz Cloud
- FAISS
- ChromaDB

> [Example code](https://github.com/zilliztech/GPTCache/blob/main/examples/data_manager/vector_store.py)

**Custom Store**

First, you need to implement two interfaces, namely [`CacheStorage`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/base.py) and [`VectorBase`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/base.py), and then create the corresponding data manager through the `get_data_manager` method.

Reference: [CacheStorage sqlalchemy](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/scalar_data/sqlalchemy.py) [VectorBase Faiss](https://github.com/zilliztech/GPTCache/blob/main/gptcache/manager/vector_data/faiss.py)

```python
from gptcache import cache
from gptcache.manager import get_data_manager

data_manager=get_data_manager(cache_base=CustomCacheStore(), vector_base=CustomVectorStore())
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

- **config**: includes cache-related configurations, which currently consist of the following: `log_time_func`, `similarity_threshold`.

  - log_time_func: The function logging time-consuming operations currently detects `embedding` and `search` functions.
  - similarity_threshold

- **next_cache**: This points to the next cache object, which is useful for implementing multi-level cache functions.

  ```python
  from gptcache import cache, Cache
  from gptcache.manager import get_data_manager 
  
  bak_cache = Cache()
  bak_data_file = "data_map_bak.txt"
  bak_cache.init(data_manager=get_data_manager(data_path=bak_data_file))
  
  cache.init(data_manager=get_data_manager(), next_cache=bak_cache)
  ```
  
## Request cache parameter customization

- **cache_obj**: customize request cache, use global variable cache by default.

```python
onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
one_cache = Cache()
one_cache.init(embedding_func=onnx.to_embeddings,
               data_manager=data_manager,
               evaluation_func=pair_evaluation,
               config=Config(
                   similarity_threshold=1,
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

## How to run with session

Session can isolate the context of each connection, and can also filter the results after recall, and if not satisfied will re-request rather than return the cache results directly. 

First we need to initialize the cache:

```python
from gptcache import cache

cache.init()
cache.set_openai_key()
```

Then we can set the session parameter for each request.

### Run in `with` method

```python
from gptcache.session import Session

with Session() as session:
    response = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo',
                            messages=[
                              {
                                  'role': 'user',
                                  'content': "what's github"
                              }],
                            session=session
                          )
```

> The `with` method will delete the session data directly on exit, if you don't want to delete data in your sesion, you can run the following code but without `session.drop()`.
>
> If you are using `LangChainLLMs`, you can set the session for current LLM, e.g. `llm = LangChainLLMs(llm=OpenAI(temperature=0), session=session)`. Or you can run the llm with specific session, e.g. `llm = LangChainLLMs(llm=OpenAI(temperature=0)` , and run `llm(question, session=session)`.

### Custom Session

You can customize the `name` of the sesion, and the `check_hit_func` method to check if a hit is satisfied, which method has four parameters:

- `cur_session_id`:  the name of the current session

- `cache_session_ids`: a list of session names for caching the same content if you are using map as a data management method. Otherwise a list of session names for similar content and same answer

- `cache_question`: a list with one question which same as the you asked if you use a map as a data management method. Otherwise it is a list that is similar to the question you asked with the same answer, and it is correspondence with `cache_session_ids`

- `cache_answer`: the content of the cached answer

> The default `check_hit_func` returns `cur_session_id not in cache_session_ids`, which means that the answers returned cannot be in the same session.

In the following code,  `my_check_hit` is defined to check if the cached answer contains "GitHub", and return `True` if it does, then gptcache will continue with the subsequent evaluation operations, and if it does not contain it will return `False` and will re-run the request.

```python
from gptcache.session import Session

def my_check_hit_func(cur_session_id, cache_session_ids, cache_questions, cache_answer):
    if "GitHub" in cache_answer:
        return True
    return False
session = Session(name="my-session", check_hit_func=my_check_hit_func)

response = openai.ChatCompletion.create(
                          model='gpt-3.5-turbo',
                          messages=[
                            {
                                'role': 'user',
                                'content': "what's github"
                            }],
                          session=session
                        )
# session.drop() # Optional
```

And you can also run `data_manager.list_sessions` to list all the sessions.

## How to use GPTCache server

GPTCache now supports building a server with caching and conversation capabilities. You can start a customized GPTCache service within a few lines.

### Start server

Once you have GPTCache installed, you can start the server with following command:
```shell
$ gptcache_server -s [HOST] -p [PORT] -d [CACHE_DIRECTORY] -f [CACHE_CONFIG_FILE]
```
The args are optional:
- -s/--host: Specify the host to start GPTCache service, defaults to "0.0.0.0".
- -p/--port: Specify the port to access to the service, defaults to 8000.
- -d/--cache-dir: Specify the directory of the cache, defaults to `gptcache_data` folder.
- -f/--cache-config-file: Specify the YAML file to config GPTCache service, defaults to None.

**GPTCache server configuration**

You can config the server via a YAML file, here is an example config yaml:

```yaml
embedding:
    onnx
embedding_config:
    # Set embedding model params here
storage_config:
    data_dir:
        gptcache_data
    manager:
        sqlite,faiss
    vector_params:
        # Set vector storage related params here
evaluation: 
    distance
evaluation_config:
    # Set evaluation metric kws here
pre_function:
    get_prompt
post_function:
    first
config:
    similarity_threshold: 0.8
    # Set other config here
```
- embedding: The embedding model source, options: [How to set the **embedding** function](#How-to-set-the-embedding-function)
- embedding_config: The embedding model config, details: [Embedding Reference](https://gptcache.readthedocs.io/en/latest/references/embedding.html)
- data_dir: The cache directory.
- manager: The cache storage and vector storage.
- evaluation: The evaluation component, options: [How to set the **similarity evaluation** interface](#How-to-set-the-similarity-evaluation-interface)
- evaluation_config: The evaluation config, options: [Similarity Evaluation Reference](https://gptcache.readthedocs.io/en/latest/references/similarity_evaluation.html)
- pre_function: The pre-processing function.
- post_function: The post-processing function.
- config: The cache config, like `similarity_threshold`

**Use the docker to start the GPTCache server**

Also, you can start the service in a docker container:

- Get image from the dockerhub
    ```shell
    $ docker pull zilliz/gptcache:latest
    ```
- Run the service in a container with default port
    ```shell
    $ docker run -p 8000:8000 -it zilliz/gptcache:latest
    ```
- Run the service in a container with certain port (e.g. 8000) and config file (e.g. gptcache.yml)
    ```shell
    $ docker run -p 8000:8000 -it gptcache:v0 gptcache_server -s 0.0.0.0 -p 8000 -f gptcache.yml
    ```

**Interact with the server**

GPTCache supports two ways of interaction with the server:

- With command line:

put the data to cache

```shell
curl -X 'POST' \
  'http://localhost:8000/put' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Hi",
  "answer": "Hi back"
}'
```

get the data from the cache

```shell
curl -X 'POST' \
  'http://localhost:8000/get' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Hi"
}'
```


- With python client:

```python
 >>> from gptcache.client import Client

 >>> client = Client(uri="http://localhost:8000")
 >>> client.put("Hi", "Hi back")
 200
 >>> client.get("Hi")
 'Hi back'
 ```

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
| 0.95      | 0.12s        | 425      | 25       | 549        |
| 0.9       | 0.23s        | 804      | 77       | 118        |
| 0.8       | 0.26s        | 904      | 92       | 3          |
