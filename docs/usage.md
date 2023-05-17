# GPTCache Quick Start

GPTCache is easy to use and can reduce the latency of LLM queries by 100x in just two steps:

1. __Build your cache.__ In particular, you'll need to decide on an embedding function, similarity evaluation function, where to store your data, and the eviction policy.
2. __Choose your LLM.__ GPTCache currently supports OpenAI's ChatGPT (GPT3.5-turbo) and langchain. Langchain supports a variety of LLMs, such as Anthropic, Huggingface, and Cohere models.

### Build your **Cache**

The default interface for `Cache` is as follows:

```python
class Cache:
   def init(self,
            cache_enable_func=cache_all,
            pre_embedding_func=last_content,
            embedding_func=string_embedding,
            data_manager: DataManager = get_data_manager(),
            similarity_evaluation=ExactMatchEvaluation(),
            post_process_messages_func=first,
            config=Config(),
            next_cache=None,
            **kwargs
            ):
       self.has_init = True
       self.cache_enable_func = cache_enable_func
       self.pre_embedding_func = pre_embedding_func
       self.embedding_func = embedding_func
       self.data_manager: DataManager = data_manager
       self.similarity_evaluation = similarity_evaluation
       self.post_process_messages_func = post_process_messages_func
       self.data_manager.init(**kwargs)
       self.config = config
       self.next_cache = next_cache

```

Before creating a GPTCache, consider the following questions:

1. How will you generate embeddings for queries? (`embedding_func`)
   
    This function embeds text into a dense vector for context similarity search. GPTCache currently supports five methods for embedding context: OpenAI, Cohere, Huggingface, ONNX, and SentenceTransformers. We also provide a default string embedding method which serves as simple passthrough.
    
    For example, to use ONNX Embeddings, simply initialize your embedding function as `onnx.to_embeddings`.
    
    ```python
    data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
    
    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    cache.set_openai_key()
    ```
    
    Check out more [examples](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-embedding-function) to see how to use different embedding functions.
    
2. Where will you cache the data? (`data_manager` cache storage)
   
    The cache storage stores all scalar data such as original questions, prompts, answers, and access times. GPTCache supports a number of cache storage options, such as SQLite, MySQL, and PostgreSQL. More NoSQL databases will be added in the future.
    
3. Where will you store and search vector embeddings? (`data_manager` vector storage)
   
    The vector storage component stores and searches across all embeddings to find the most similar results semantically. GPTCache supports the use of vector search libraries such as FAISS or vector databases such as Milvus. More vector databases and cloud services will be added in the future.

    Here are some examples:

   ```python
   ## create user defined data manager
   data_manager = get_data_manager()
   ## create data manager with sqlite and faiss 
   data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=128))
   ## create data manager with mysql and milvus, max cache size is 100
   data_manager = get_data_manager(CacheBase("mysql"), VectorBase("milvus", dimension=128), max_size=100)
   ## create data manager with mysql and milvus, max cache size is 100, eviction policy is LRU
   data_manager = get_data_manager(CacheBase("mysql"), VectorBase("milvus", dimension=128), max_size=100, eviction='LRU') 
   ```
   
   Check out more [examples](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-data-manager-class) to see how to use different data managers.

4. What is the eviction policy?
   
    GPTCache supports evicting data based on cache count. You can choose to use either the LRU or FIFO policy. In the future, we plan to support additional cache policies, such as evicting data based on last access time or last write time.

5. How will you determine cache hits versus misses? (`evaluation_func`)

   The evaluation function helps to determine whether the cached answer matches the input query. It takes three input values: `user request data`, `cached data`, and `user-defined parameters`. GPTCache currently supports three types of evaluation functions: exact match evaluation, embedding distance evaluation and ONNX model evaluation.

   To enable ONNX evaluation, simply pass `EvaluationOnnx` to `similarity_evaluation`. This allows you to run any model that can be served on ONNX. We will support Pytorch, TensorRT and the other inference engines in the future.

   ```python
   onnx = EmbeddingOnnx()
   data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
   evaluation_onnx = EvaluationOnnx()
   cache.init(
       embedding_func=onnx.to_embeddings,
       data_manager=data_manager,
       similarity_evaluation=evaluation_onnx,
   )
   ```

   Check out our [examples](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-similarity-evaluation-interface) page to see how to use different similarity evaluation functions.

Users can also pass in other configuration options, such as:

- `log_time_func`: A function that logs time-consuming operations such as `embedding` and `search`.
- `similarity_threshold`: The threshold used to determine when embeddings are similar to each other.

### **Chose your adapter**

GPTCache currently supports two LLM adapters: OpenAI and Langchain.

With the OpenAI adapter, you can specify the model you want to use and generate queries as a user role.

```python
cache.init()
cache.set_openai_key()

question = "what's github"
answer = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
print(answer)
```

Here's an example that utilizes OpenAI's stream response API:

```python
from gptcache.cache import get_data_manager
from gptcache.core import cache, Cache
from gptcache.adapter import openai

cache.init(data_manager=get_data_manager())
os.environ["OPENAI_API_KEY"] = "API KEY"
cache.set_openai_key()

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)

# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk['choices'][0]['delta']  # extract the message
    collected_messages.append(chunk_message)  # save the message

full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
```

If you want to use other LLMs, the Langchain adapter provides support a standard interface to connect with Langchain-supported LLMs.

```python
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_cache = Cache()
llm_cache.init(
    pre_embedding_func=get_prompt,
    post_process_messages_func=postnop,
)

cached_llm = LangChainLLMs(llm)
answer = cached_llm(question, cache_obj=llm_cache)
```

We plan to support other models soon, so any contributions or suggestions are welcome.

### Other request parameters

**cache_obj**: Customize the request cache. Use this if you want to make the cache a singleton.

```python
onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
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

**cache_context**: Custom cache functions can be passed separately for each of the request.

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
      "get_scalar_data": {},
      "evaluation_func": {},
    }
)
```

**cache_skip**: This option allows you to skip the cache search, but still store the results returned by the LLM model. 

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

**session:** Specify the sesion of the current request, you can also set some rules to check if the session hits the cache, see this [example](https://github.com/zilliztech/GPTCache/tree/main/examples#How-to-run-with-session) for more details.

```python
from gptcache.session import Session

session = Session(name="my-session")
question = "what do you think about chatgpt"

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question}
    ],
    session=session
)
```

**temperature**: You can always pass a parameter of temperature with value between 0 and 2 to control randomity of output. A higher value of temperature like 0.8 will make the output more random. A lower value like 0.2 makes the output more coherent given the same input.

> The range of `temperature` is [0, 2], default value is 0.0.
> 
> A higher temperature means a higher possibility of skipping cache search and requesting large model directly.
> When temperature is 2, it will skip cache and send request to large model directly for sure. When temperature is 0, it will search cache before requesting large model service.
> 
> The default `post_process_messages_func` is `temperature_softmax`. In this case, refer to [API reference](https://gptcache.readthedocs.io/en/latest/references/processor.html#module-gptcache.processor.post) to learn about how `temperature` affects output.

```python
import time

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter import openai

cache.set_openai_key()

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )
# cache.config = Config(similarity_threshold=0.2)

question = "what's github"

for _ in range(3):
    start = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 1.0,  # Change temperature here
        messages=[{
            "role": "user",
            "content": question
        }],
    )
    print("Time elapsed:", round(time.time() - start, 3))
    print("Answer:", response["choices"][0]["message"]["content"])
```

### Use GPTCache server

GPTCache now supports building a server with caching and conversation capabilities. You can start a customized GPTCache service within a few lines. Here is a simple example to show how to build and interact with GPTCache server. For more detailed information, arguments, parameters, refer to [this](https://github.com/zilliztech/gpt-cache/tree/main/examples).

**Start server**

Once you have GPTCache installed, you can start the server with following command:
```shell
$ gptcache_server -s 127.0.0.1 -p 8000
```

**Start server with docker**

```shell
$ docker pull zilliz/gptcache:latest
$ docker run -p 8000:8000 -it zilliz/gptcache:latest
```

**Interact with the server**

GPTCache supports two ways of interaction with the server:

- With command line:
    ```shell
    $ curl -X PUT -d "receive a hello message" "http://localhost:8000?prompt=hello"
    $ curl -X GET  "http://localhost:8000?prompt=hello"
    "receive a hello message"
    ```
- With python client:
    ```python
    >>> from gptcache import Client

    >>> client = Client(uri="http://localhost:8000")
    >>> client.put("Hi", "Hi back")
    200
    >>> client.get("Hi")
    'Hi back'
    ```
