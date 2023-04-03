# Cache Usage

## Cache init parameter introduction

```python
class Config:
    def __init__(self,
                 log_time_func=None,
                 enable_report_time=True,
                 similarity_threshold=0.5,
                 similarity_positive=True,
                 ):
        self.log_time_func = log_time_func
        self.enable_report_time = enable_report_time
        self.similarity_threshold = similarity_threshold
        self.similarity_positive = similarity_positive

class Cache:
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager = None
        self.evaluation_func = None
        self.post_process_messages_func = None
        self.config = None
        self.report = Report()
        self.next_cache = None

    def init(self,
             cache_enable_func=cache_all,
             pre_embedding_func=last_content,
             embedding_func=string_embedding,
             data_manager: DataManager = get_data_manager("map"),
             evaluation_func=absolute_evaluation,
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
        self.evaluation_func = evaluation_func
        self.post_process_messages_func = post_process_messages_func
        self.data_manager.init(**kwargs)
        self.config = config
        self.next_cache = next_cache
```

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

- **embedding_func**: Embed the text into a dense vector for similarity search.

    The function has two parameters: the preprocessed string and parameters reserved for user customization. To acquire these parameters, a similar method to the one above is used: `kwargs.get("embedding_func", {})`.

    ```python
    def to_embeddings(data, **kwargs):
        return data
    ```

    ```python
    class Towhee:
        # english model: paraphrase-albert-small-v2
        # chinese model: uer/albert-base-chinese-cluecorpussmall
        def __init__(self, model="paraphrase-albert-small-v2"):
            self._pipe = (
                pipe.input('text')
                    .map('text', 'vec',
                         ops.sentence_embedding.transformers(model_name=model))
                    .map('vec', 'vec', ops.towhee.np_normalize())
                    .output('text', 'vec')
            )
            self.__dimension = len(self._pipe("foo").get_dict()['vec'])
    
        def to_embeddings(self, data, **kwargs):
            emb = self._pipe(data).get_dict()['vec']
            return np.array(emb).astype('float32')
    
        def dimension(self):
            return self.__dimension
    ```

    Note that if you intend to use the model, it should be packaged with a class. The model will be loaded when the object is created to avoid unnecessary loading when not in use. This also ensures that the model is not loaded multiple times during program execution.

- **data_manager**: which includes searching, saving, or evicting data.

    Currently, there are three creation methods available, namely `get_data_manager`, `get_ss_data_manager`, and `get_si_data_manager`.

    1. *get_data_manager*, retrieves the corresponding data manager by name. Currently, the following are supported: `map`, `scalar_vector`, and `scalar_vector_index`.
    
    ```python
    # param list
    # data_path: data persistence path
    # max_size: maximum amount of cached data, default value: 100
  
    data_manager=get_data_manager("map")
    cache.init(data_manager=data_manager)
    ```
  
    ```python
    # param list
    # max_size: maximum amount of cached data, default value: 1000
    # clean_size: the maximum number of caches has been reached, and the number of cache is cleared. default value: 1000 * 0.2
  
    data_manager=get_data_manager("scalar_vector", scalar_store=Sqlite(), vector_store=Milvus())
    cache.init(data_manager=data_manager)
    ```
    
    ```python
    # param list
    # max_size: maximum amount of cached data, default value: 1000
    # clean_size: the maximum number of caches has been reached, and the number of cache is cleared. default value: 1000 * 0.2
  
    data_manager=get_data_manager("scalar_vector_index", scalar_store=Sqlite(), vector_index=Milvus())
    cache.init(data_manager=data_manager)
    ```
  
    Note that the latter two methods are primarily intended for users who want to access custom storage methods. The provided example does not include parameters for creating corresponding storage objects. For instance, when creating a vector data object, it is typically necessary to specify its dimension.

    2. *get_ss_data_manager*, retrieves the corresponding data manager for scalar and vector databases.

    ```python
    # common param list
    # max_size: like above
    # clean_size: like above
  
    # sqlite param list
    # sqlite_path: sqlite database path
    # eviction_strategy: cache eviction strategy, Cache elimination strategy, which can be set to `least_accessed_data` and `oldest_created_data`
  
    # milvus param list
    # dimension: vector dimension, indispensable parameter.
    # top_k: number of search results
    # Other milvus creation parameters are set through kwargs, like: host, port, user, password, is_https, collection_name
  
    data_manager = get_ss_data_manager("sqlite", "milvus", dimension=d)
    cache.init(data_manager=data_manager)
    ```

    3. *get_ss_data_manager*, retrieves the corresponding data manager for scalar databases and vector index, this will be deprecated.

    ```python
    # common param list
    # max_size: like above
    # clean_size: like above
  
    # sqlite param list
    # sqlite_path: like above
    # eviction_strategy: like above
  
    # faiss param list
    # index_path: faiss index path
    # dimension: vector dimension, indispensable parameter.
    # top_k: number of search results
  
    data_manager = get_si_data_manager("sqlite", "faiss", dimension=d)
    cache.init(data_manager=data_manager)
    ```
  
- **evaluation_func**: evaluate similarity by judging the quality of cached answers.

  The function takes three input values, namely `user request data`, `cache data`, and `user-defined data`. The last parameter, by using `kwargs.get("evaluation_func", {})`, is reserved for users and can be used in the same way as `pre_process`.

  ```python
  # param
  rank = chat_cache.evaluation_func({
              "question": pre_embedding_data,
              "embedding": embedding_data,
          }, {
              "question": cache_question,
              "answer": cache_answer,
              "search_result": cache_data,
          }, extra_param=context.get('evaluation_func', None))
  ```
  
  ```python
  # search distance
  def pair_evaluation(src_dict, cache_dict, **kwargs):
      distance, _ = cache_dict["search_result"]
      return distance
  ```
  
  ```python
  class Towhee:
      def __init__(self):
          self._pipe = (
              pipe.input('text', 'candidate')
                  .map(('text', 'candidate'), 'similarity', ops.towhee.albert_duplicate())
                  .output('similarity')
          )
  
      # WARNING: the model cannot evaluate text with more than 512 tokens
      def evaluation(self, src_dict, cache_dict, **kwargs):
          try:
              src_question = src_dict["question"]
              cache_question = cache_dict["question"]
              return self._pipe(src_question, [cache_question]).get_dict()['similarity'][0]
          except Exception:
              return 0
  ```
  
  Note that if you intend to use the model, it should be packaged with a class. The model will be loaded when the object is created to avoid unnecessary loading when not in use. This also ensures that the model is not loaded multiple times during program execution.

- **config**: includes cache-related configurations, which currently consist of the following: `log_time_func`, `similarity_threshold`, and `similarity_positive`.

  - log_time_func: The function logging time-consuming operations currently detects `embedding` and `search` functions.
  - similarity_threshold
  - similarity_positive: When set to `True`, a higher value indicates a higher degree of similarity. When set to `False`, a lower value indicates a higher degree of similarity.

- **next_cache**: This points to the next cache object, which is useful for implementing multi-level cache functions.

  ```python
  bak_cache = Cache()
  bak_data_file = dir_name + "/data_map_bak.txt"
  bak_cache.init(data_manager=get_data_manager("map", data_path=bak_data_file))
  data_file = dir_name + "/data_map.txt"
  
  cache.init(data_manager=get_data_manager("map"),
             next_cache=bak_cache)
  ```

## Request cache parameter customization

- **cache_obj**: customize request cache, use global variable cache by default.

```python
towhee = Towhee()
data_manager = get_si_data_manager("sqlite", "faiss", dimension=towhee.dimension())
one_cache = Cache()
one_cache.init(embedding_func=towhee.to_embeddings,
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
```