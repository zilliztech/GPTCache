# GPTCache Quick Start

GPTCache is easy to understand and can boost your LLM API speed by 100x in just two steps:

1. Build your cache, which includes deciding on embedding functions, similarity evaluation functions, where to store the data, and eviction configurations.
2. Choose your adapter. GPTCache currently supports the OpenAI chatGPT interface and langchain interface. Langchain supports a variety of LLMs, such as Anthropic, Llama-cpp, Huggingface hub, and Cohere.

### Build your **Cache**

The interface for initializing the cache looks like the following:

```
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

Before creating GPTCache, consider the following questions:

1. How will you generate an embedding for the query? (`embedding_func`)
   
    This function embeds text into a dense vector for context similarity search. GPTCache currently supports five methods for embedding context: OpenAI, Cohere, Hugging Face API, ONNX model serving, and SentenceTransformers.
    
    For instance, to use ONNX Embeddings, simply initialize your embedding function to `onnx.to_embeddings`.
    
    ```
    data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
    
    cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
              )
    cache.set_openai_key()
    ```
    
    Check out more [examples](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-embedding-function) to see how to use different embedding functions.
    
2. Where will you cache the data? (`data_manager cache storage`)
   
    The cache storage stores all scalar data such as original questions, prompts, answers, and access times. GPTCache supports various cache storage options, such as SQLite, MySQL, or PostgreSQL, and more NoSQL databases will be added in the future.
    
3. Where will you store and search vector embeddings? (`data_manager vector storage`)
   
    The vector storage stores all the embeddings and searches for the most similar results semantically. GPTCache supports the use of vector search libraries such as FAISS or vector databases such as Milvus, and more vector databases and cloud services will be added in the future.
    
4. When will the cache evict?
   
    GPTCache supports evicting data based on cache count. Users can choose to use either the LRU or FIFO policy. In the future, we plan to support additional cache policies, such as evicting data based on last access time or last write time.
    
    Here are some examples to create data_manager:

   ```
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

 5.  How will you determine whether it's a hit or miss? (`evaluation_func`)

   The evaluation function helps to determine whether the cached answer is similar or not. It takes three input values: `user request data`, `cached data`, and `user-defined parameters`. GPTCache now supports three types of evaluation: exact match evaluation, embedding distance evaluation and ONNX model evaluation.

   To enable ONNX evaluation, simply pass EvaluationOnnx to similarity_evaluation. This allows you to run any model that can be served on ONNX. We will support pytorch, tensorRT and the other inference engine in the future.

   ```
   onnx = EmbeddingOnnx()
   data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
   evaluation_onnx = EvaluationOnnx()
   cache.init(embedding_func=onnx.to_embeddings,
                  data_manager=data_manager,
                  similarity_evaluation=evaluation_onnx,
                  )
   ```

   Check out more [examples](https://github.com/zilliztech/gpt-cache/tree/main/examples#How-to-set-the-similarity-evaluation-interface) to see how to use different similarity evaluation functions.

Users can also pass in other configuration options , such as:

- `log_time_func`: A function that logs time-consuming operations such as `embedding` and `search`.
- `similarity_threshold`: The threshold used to determine when embeddings are similar to each other.

### **Chose your adaptor**

GPTCache now supports two LLM adapters: OpenAI and Langchain.

With the OpenAI adapter, you can specify the model you want to use and generate queries as a user role.

```
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

if you want to utilize stream response API in openAI SDK:

```
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

If you want to connect with other LLMs, the Langchain Adaptor provides support for a standard interface for all of them.

```
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

We are planning to support other large language models in the future,  any contributions or suggestions would be highly welcomed.

### Other request parameters

**cache_obj**: Customize the request cache. Use this if you want to make the cache a singleton.

```
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

```
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

```
question = "what do you think about chatgpt"

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question}
    ],
    cache_skip=True
)
```