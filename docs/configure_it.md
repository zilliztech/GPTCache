# How to better configure your cache

**Last update time**: 2023.6.26

**Latest version**: v0.1.32

Before reading the following content, you need to understand the basic composition of GPTCache, you need to finish reading:

- [GPTCache README](https://github.com/zilliztech/GPTCache)
- [GPTCache Quick Start](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md)

## Introduction to GPTCache initialization

GPTCache **core components** include:

- pre-process func
- embedding
- data manager
  - cache store
  - vector store
  - object store (optional, multi-model)
- similarity evaluation
- post-process func

The above core components need to be set when similar caches are initialized, and of course most of them have default values. In addition to these, there are additional parameters, including:

- **config**, some configurations of the cache, such as similarity thresholds, parameter values of some specific preprocessing functions, etc.;
- **next_cache**, can be used to set up a multi-level cache. 
  
    For example, there are two GPTCaches, L1 and L2, where L1 sets L2 as the next cache during initialization.

    When accepting a user request, if the L1 cache misses, it will go to the L2 cache to find it.

    If the L2 also misses, it will call the LLM, and then store the results in the L1 and L2 caches. 

    If the L2 hits, the cached result will be stored in the L1 cache

The above is the basic description of all initialization parameters.

In GPTCache lib, there is a global cache object. If the llm request does not set the cache object, this global object is used.

There are currently **three** methods of initializing the cache, namely:

1. The `init` method of the `Cache` class defaults to exact key matching, which is a simple map cache, that is:

```
def init(
    self,
    cache_enable_func=cache_all,
    pre_func=last_content,
    embedding_func=string_embedding,
    data_manager: DataManager = get_data_manager(),
    similarity_evaluation=ExactMatchEvaluation(),
    post_func=temperature_softmax,
    config=Config(),
    next_cache=None,
  ):
  pass
```

2. The `init_similar_cache` method in the api package defaults to similar matching of onnx+sqlite+faiss

```
def init_similar_cache(
    data_dir: str = "api_cache",
    cache_obj: Optional[Cache] = None,
    pre_func: Callable = get_prompt,
    embedding: Optional[BaseEmbedding] = None,
    data_manager: Optional[DataManager] = None,
    evaluation: Optional[SimilarityEvaluation] = None,
    post_func: Callable = temperature_softmax,
    config: Config = Config(),
  ):
  pass
```

3. The `init_similar_cache_from_config` in the api package initializes the cache through the yaml file, and the default is fuzzy matching of onnx+sqlite+faiss, more details: [GPTCache server configuration](https://github.com/zilliztech/GPTCache/tree/main/examples#start-server)

```
def init_similar_cache_from_config(config_dir: str, cache_obj: Optional[Cache] = None):
  pass
```

## Pre-Process function

The preprocessing function is mainly used to obtain user question information from the user llm request parameter list, assemble this part of information into a string and return it. The return value is the input of the embedding model.

It is worth noting that **different llms need to use different preprocessing functions**, because the request parameter list of each llm is inconsistent. And the parameter names containing user problem information are also different.

Of course, if you want to use different pre-processing processes according to other llm parameters of the user, this is also possible.

The definition of the preprocessing function receives two parameters, and the return value can be one or two.

```python
def foo_pre_process_func(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    pass
```

Among them, `data` is the list of user parameters, and `params` is some additional parameters, such as cache config, which can be obtained through `params.get("cache_config", None)`.

If there is no special requirement, the function can return a value, which is used for the input of embedding and the key of the current request cache.

Of course, two values can also be returned, the first one is used as the key of the current request cache, and the second one is used as the input of embdding, which is currently mainly used to **handle long openai chat conversations**. In the case of a long dialogue, the first return value is the user's original long dialogue, and only simple dialogue string splicing is performed, and the second return value is to extract the key information of the long dialogue through some models, shortening the embedding input.

**Currently available preprocessing functions:**

all source code reference: [processor/pre](https://github.com/zilliztech/GPTCache/blob/main/gptcache/processor/pre.py)

all preprocessing api reference: [gptcache.processor.pre](https://gptcache.readthedocs.io/en/latest/references/processor.html#module-gptcache.processor.pre)

If you are confused about the role of the following preprocessing functions, **you can check the api reference**, which contains simple function examples.

### openai chat complete

- last_content: get the last content of the message list.
- last_content_without_prompt: get the last content of the message list without prompts content. It needs to be used with the `prompts` parameter in [Config](https://gptcache.readthedocs.io/en/latest/references/gptcache.html#module-gptcache.config). If it is not set, it will have the same effect as last_content.
- last_content_without_template: get the last content's template values of the message list without template content. The functionality is similar to the previous one, but it can handle more complex templates. The above is only a simple judgment through the string, that is, the user's prompt must be continuous. But `last_content_without_template` can support string template type, please refer to api reference for specific usage.
- all_content: simply concat the contents of the messages list in the user request.
- concat_all_queries: concat the content and role info of the message list.
- context_process: to deal with long dialogues in openai, the core is to compress the dialogue through some methods, and extract the core content of the dialogue as the key of the cache.
  - summarization_context, compress dialogue content through the summary model, api reference: [processor.context.summarization_context](https://gptcache.readthedocs.io/en/latest/references/processor.html#module-gptcache.processor.context.summarization_context)
  - selective_context, select parts of a dialog by model, api reference: [processor.context.selective_context](https://gptcache.readthedocs.io/en/latest/references/processor.html#module-gptcache.processor.context.selective_context)
  - concat_context, concat all parts of a dialog, which is easy to handle with rwkv embedding, api reference: [processor.context.concat_context](https://gptcache.readthedocs.io/en/latest/references/processor.html#module-gptcache.processor.context.concat_context)

### langchain llm

- get_prompt: get the `prompt` of the llm request params.

### langchain chat llm

- get_messages_last_content: get the last content of the llm request `message` object array.

### openai image

- get_prompt: get the `prompt` of the llm request params.

### openai audio

- get_file_name: get the file name of the llm request params
- get_file_bytes: get the file bytes of the llm request params

### openai moderation

- get_openai_moderation_input: get the input param of the openai moderation request params

### llama

- get_prompt: get the `prompt` of the llm request params.

### replicate (image -> text, image and text -> text)

- get_input_str: get the image and question str of the llm request params
- get_input_image_file_name: get the image file name of the llm request params

### stable diffusion

- get_prompt: get the `prompt` of the llm request params.

### minigpt4

- get_image_question: get the image and question str of the llm request params
- get_image: get the image of the llm request params

### dolly

- get_inputs: get the inputs of the llm request params

NOTE: **For different llm, different preprocessing functions should be selected** when the cache is initialized. If not, you can choose to customize.

## Embedding

Convert the input into a multidimensional array of numbers, which are classified according to the input type.

Whether the cache is accurate or not, the choice of embedding model is more important. **A few points worth noting**: the language supported by the model, and the number of tokens supported by the model.
In addition, generally speaking, under certain computer resources, large models are more accurate, but time-consuming; small models run faster, but are less accurate.

all embedding api reference: [embedding api](https://gptcache.readthedocs.io/en/latest/references/embedding.html)

### text

- Onnx: small, only supports 512token, and only supports English
- Huggingface: default [Distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased), Chinese [uer/albert-base-chinese-cluecorpussmall](https://huggingface.co/uer/albert-base-chinese-cluecorpussmall), more models: [huggingface models](https://huggingface.co/models?sort=downloads)
- SBERT: optional model list reference: [sbert Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- OpenAI: openai embedding api server, more details: [openai embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- Cohere: cohere embedding api server, more details: [cohere embed](https://docs.cohere.com/reference/embed)
- LangChain: langchain text embedding models, more details: [langchain text embedding models](https://langchain-langchain.vercel.app/docs/modules/data_connection/text_embedding/), [GPTCache langchain embedding usage](https://gptcache.readthedocs.io/en/latest/references/embedding.html#module-gptcache.embedding.langchain)
- Rwkv: rwkv text embedding models, more details: [huggingface transformers rwkv](https://huggingface.co/docs/transformers/model_doc/rwkv)
- PaddleNLP: easy-to-use and powerful NLP library, more details: [PaddleNLP Transformer models](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html)
- UForm: multi-modal transformers library, more details: [ufrom usage](https://unum-cloud.github.io/uform/)
- FastText: library for fast text representation and classification, more details: [fastText](https://github.com/facebookresearch/fastText#models)

### audio

- Data2VecAudio: huggingface audio embedding model, more details: [huggingface data2vec-audio](https://huggingface.co/models?sort=downloads&search=data2vec-audio)

### image

- Timm: huggingface image embedding model, more details: [huggingface timm](https://huggingface.co/docs/timm/quickstart)
- ViT: huggingface vit image embedding model, more details: [huggingface vit](https://huggingface.co/models?sort=downloads&search=vit)

**NOTE**: you need to select the appropriate embedding model according to the data type, and you also need to look at the language supported by embedding.

## Data Manager

For the similar cache of text, only cache store and vector store are needed. If it is a multi-modal cache, object store is additionally required. The choice of storage is not related to the llm type, but it should be noted that the vector dimension needs to be set when using the vector store.

### cache store

- sqlite
- duckdb
- mysql
- mariadb
- sqlserver
- oracle
- postgresql

### vector store

- milvus
- faiss
- chromadb
- hnswlib
- pgvector
- docarray
- usearch
- redis

### object store

- local
- s3

### how to get a data manager

- Use factory to get it by the store name

`vector_params` is the parameter required to build the vector store；

`scalar_params` is the parameter required to build the cache store；

```python
from gptcache.manager import manager_factory

data_manager = manager_factory("sqlite,faiss", data_dir="./workspace", scalar_params={}, vector_params={"dimension": 128})
```

- Combining each store object through get_data_manager method

```python
from gptcache.manager import get_data_manager, CacheBase, VectorBase

data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
```

**Note that** each store has more initialization parameters, you can reference the store's constructor method by the [store api reference](https://gptcache.readthedocs.io/en/latest/references/manager.html).

## Similarity Evaluation

If you want the cache to play a better role, in addition to embedding and vector engines, appropriate similarity evaluation is also very critical.

The similarity evaluation is mainly: evaluate the recalled cache data according to the current user's llm request, and obtain a float value. The easiest way is to use the embedding distance. Of course, there are other methods, such as using a model to judge the similarity of two problems.

The following are similar evaluation components that already exist.

1. SearchDistanceEvaluation, vector search distance, simple, fast, but not very accurate
2. OnnxModelEvaluation, use the model to compare the degree of correlation between the two questions. The small model only supports 512token, which is more accurate than the distance
3. NumpyNormEvaluation, calculate the distance between the two embedding vectors of the llm request and the cache data, which is fast and simple, and the accuracy is almost the same as the distance
4. KReciprocalEvaluation, use the K-reprocical algorithm to calculate the similarity for reranking, and recall multiple cache data for comparison. It needs to be recalled many times, which is more time-consuming and relatively more accurate. For more information, refer to the api reference
5. CohereRerankEvaluation, use the cohere rerank api server, more accurate, at a cost, more details: [cohere rerank](https://docs.cohere.com/reference/rerank-1)
6. SequenceMatchEvaluation, sequence matching, suitable for multiple rounds of dialogue, separates each round of dialogue for similar evaluation, and then obtains the final score through the proportion
7. TimeEvaluation, evaluate by cache creation time, avoid using stale cache
8. SbertCrossencoderEvaluation, use the sbert model for rerank evaluation, **which is currently the best similarity evaluation found**

More detailed usage reference [api doc](https://gptcache.readthedocs.io/en/latest/references/similarity_evaluation.html)

Of course, if you want to get a better Similarity Evaluation, **you need to customize it according to the scene**, such as assembling the existing Similarity Evaluation. If you want to get better caching effect in long conversations, you may need to assemble SequenceMatchEvaluation, TimeEvaluation, TimeEvaluation, of course there may be a better way.

## Post-Process function

Post-processing is mainly to obtain the final answer to user questions based on all cached data that meet the similarity threshold. One of them can be selected according to a certain strategy in the cached data list, or the model can be used to fine-tune these answers, so that similar questions can have different answers.

Currently Existing Postprocessing Functions：

1. temperature_softmax, select according to the softmax strategy, which can ensure that the obtained cached answer has a certain randomness
2. first, get the most similar cached answer
3. random, randomly fetch a similar cached answer

## Recommended by newcomers

MOTE: different llm corresponds to different **preprocessing functions**, which need to be adjusted according to your needs !!!

### beginner level

Want to experience the function of GPTCache, use the simplest combination: `onnx embedding + (sqlite + faiss) data manager + distance similarity evaluation`

<details>

<summary> <strong>english version</strong> </summary>

```python
import time

from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import last_content

init_similar_cache(pre_func=last_content)

questions = [
    "what's github",
    "can you explain what GitHub is",
    "can you tell me more about GitHub",
    "what is the purpose of GitHub",
]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}],
    )
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response["choices"][0]["message"]["content"]}\n')
```

```text
# console output

huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Question: what's github
Time consuming: 12.23s
Answer: GitHub is a web-based platform that uses Git for version control. It provides developers with a collaborative environment in which they can store and share their code, manage projects, track issues, and build software. GitHub also provides a range of features for code collaboration and review, including pull requests, forks, and merge requests, that enable users to work together on code development and share their work with a wider community. GitHub is widely used by businesses, open-source communities, and individual developers around the world.

Question: can you explain what GitHub is
Time consuming: 0.64s
Answer: GitHub is a web-based platform that uses Git for version control. It provides developers with a collaborative environment in which they can store and share their code, manage projects, track issues, and build software. GitHub also provides a range of features for code collaboration and review, including pull requests, forks, and merge requests, that enable users to work together on code development and share their work with a wider community. GitHub is widely used by businesses, open-source communities, and individual developers around the world.

Question: can you tell me more about GitHub
Time consuming: 0.21s
Answer: GitHub is a web-based platform that uses Git for version control. It provides developers with a collaborative environment in which they can store and share their code, manage projects, track issues, and build software. GitHub also provides a range of features for code collaboration and review, including pull requests, forks, and merge requests, that enable users to work together on code development and share their work with a wider community. GitHub is widely used by businesses, open-source communities, and individual developers around the world.

Question: what is the purpose of GitHub
Time consuming: 0.24s
Answer: GitHub is a web-based platform that uses Git for version control. It provides developers with a collaborative environment in which they can store and share their code, manage projects, track issues, and build software. GitHub also provides a range of features for code collaboration and review, including pull requests, forks, and merge requests, that enable users to work together on code development and share their work with a wider community. GitHub is widely used by businesses, open-source communities, and individual developers around the world.

```

</details>

<details>

<summary> <strong>chinese version</strong> </summary>

If the question is Chinese, you need to use other embedding models, here we use the model on huggingface.

```python
import time

from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Huggingface
from gptcache.processor.pre import last_content

huggingface = Huggingface(model="uer/albert-base-chinese-cluecorpussmall")
init_similar_cache(pre_func=last_content, embedding=huggingface)

questions = ["什么是Github", "你可以解释下什么是Github吗", "可以告诉我关于Github一些信息吗"]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}],
    )
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response["choices"][0]["message"]["content"]}\n')
```

```text
# console output

Some weights of the model checkpoint at uer/albert-base-chinese-cluecorpussmall were not used when initializing AlbertModel: ['predictions.decoder.bias', 'predictions.LayerNorm.bias', 'predictions.bias', 'predictions.dense.bias', 'predictions.LayerNorm.weight', 'predictions.dense.weight', 'predictions.decoder.weight']
- This IS expected if you are initializing AlbertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing AlbertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of AlbertModel were not initialized from the model checkpoint at uer/albert-base-chinese-cluecorpussmall and are newly initialized: ['albert.pooler.weight', 'albert.pooler.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
2023-06-27 18:05:20,233 - 140704448365760 - connectionpool.py-connectionpool:812 - WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /v1/chat/completions
Question: 什么是Github
Time consuming: 24.09s
Answer: GitHub是一个面向开源及私有软件项目的托管平台，因为只支持git（一个分布式版本控制系统）作为唯一的版本库格式进行托管，故名GitHub。GitHub于2008年4月10日正式上线，除了目前，GitHub已经成为了世界上最大的开源社区和开源软件开发平台之一。

Question: 你可以解释下什么是Github吗
Time consuming: 0.49s
Answer: GitHub是一个面向开源及私有软件项目的托管平台，因为只支持git（一个分布式版本控制系统）作为唯一的版本库格式进行托管，故名GitHub。GitHub于2008年4月10日正式上线，除了目前，GitHub已经成为了世界上最大的开源社区和开源软件开发平台之一。

Question: 可以告诉我关于Github一些信息吗
Time consuming: 0.12s
Answer: GitHub是一个面向开源及私有软件项目的托管平台，因为只支持git（一个分布式版本控制系统）作为唯一的版本库格式进行托管，故名GitHub。GitHub于2008年4月10日正式上线，除了目前，GitHub已经成为了世界上最大的开源社区和开源软件开发平台之一。
```

</details>

### standard level

Understand the initialization methods of all caches, and try different components to better match your usage scenarios.

**TIPS:**
1. Different llms require different preprocessing functions, and the usage scenarios of llm also need to be considered
2. The number of tokens and languages supported when using the model, such as embedding and similarity evaluation
3. Don’t forget to pass the dimension parameter during the vector database initialization process
4. There are **many examples** in the source code, which can be found in the **bootcamp/example/test** directory
5. If there are multiple llms in a program that need to use cache, multiple Cache objects need to be created

<details>

<summary> <strong>Here is an example where each component does not use default values.</strong> </summary>

```python
import time

from gptcache import Cache, Config
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.post import random_one
from gptcache.processor.pre import last_content
from gptcache.similarity_evaluation import OnnxModelEvaluation

openai_complete_cache = Cache()
encoder = Onnx()
sqlite_faiss_data_manager = manager_factory(
    "sqlite,faiss",
    data_dir="openai_complete_cache",
    scalar_params={
        "sql_url": "sqlite:///./openai_complete_cache.db",
        "table_name": "openai_chat",
    },
    vector_params={
        "dimension": encoder.dimension,
        "index_file_path": "./openai_chat_faiss.index",
    },
)
onnx_evaluation = OnnxModelEvaluation()
cache_config = Config(similarity_threshold=0.75)

init_similar_cache(
    cache_obj=openai_complete_cache,
    pre_func=last_content,
    embedding=encoder,
    data_manager=sqlite_faiss_data_manager,
    evaluation=onnx_evaluation,
    post_func=random_one,
    config=cache_config,
)

questions = [
    "what's github",
    "can you explain what GitHub is",
    "can you tell me more about GitHub",
    "what is the purpose of GitHub",
]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        cache_obj=openai_complete_cache,
    )
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response["choices"][0]["message"]["content"]}\n')
```

```text
# console output

huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Question: what's github
Time consuming: 24.73s
Answer: GitHub is an online platform used primarily for version control and coding collaborations. It's used by developers to store, share and manage their codebase. It allows users to collaborate on projects with other developers, track changes, and manage their code repositories. It also provides several features such as pull requests, code review, and issue tracking. GitHub is widely used in the open source community and is considered as an industry standard for version control and software development.

Question: can you explain what GitHub is
Time consuming: 0.61s
Answer: GitHub is an online platform used primarily for version control and coding collaborations. It's used by developers to store, share and manage their codebase. It allows users to collaborate on projects with other developers, track changes, and manage their code repositories. It also provides several features such as pull requests, code review, and issue tracking. GitHub is widely used in the open source community and is considered as an industry standard for version control and software development.

Question: can you tell me more about GitHub
Time consuming: 33.38s
Answer: GitHub is a web-based hosting service for version control using Git. It is used by developers to collaborate on code from anywhere in the world. It allows developers to easily collaborate on projects, keep track of changes to code, and work together on large codebases. GitHub provides a comprehensive platform for developers to build software together, making it easier to track changes, test and deploy code, and manage project issues. It also hosts millions of open-source projects, making it a valuable resource for developers looking to learn from others’ code and contribute to the open-source community. Additionally, non-developers can use GitHub to store and share documents, create and contribute to wikis, and track projects and issues. GitHub is a key tool in modern software development and has come to be an essential part of the software development process.

Question: what is the purpose of GitHub
Time consuming: 0.32s
Answer: GitHub is an online platform used primarily for version control and coding collaborations. It's used by developers to store, share and manage their codebase. It allows users to collaborate on projects with other developers, track changes, and manage their code repositories. It also provides several features such as pull requests, code review, and issue tracking. GitHub is widely used in the open source community and is considered as an industry standard for version control and software development.

```

It can be found that the third problem does not hit the cache, this is because of the use of OnnxModelEvaluation. Using the model for similarity evaluation can improve the quality of cached answers, but it may also lead to a decrease in the hit rate of the cache, because it is possible to filter some that could have been cached, but the model thinks they are not similar.

**Therefore, the choice of components needs to be selected according to your own needs.**

WARNING: If it is a custom `cache`, you need to add the `cache_obj` parameter to specify the cache object when llm requests.

```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
    cache_obj=openai_complete_cache,
)
```

</details>

### professional level

Understand the source code of GPTCache, be familiar with the permission logic, and customize or create components according to your own needs.

According to the current usage, the main conditions to determine the cache quality are:
1. Preprocessing function, because the return value of the function will be used as the input of embedding
2. Embedding model
3. Vector Store
4. Similarity evaluation, using the rerank model for similar evaluation

The GPTCache product we use internally, according to the existing data, found that the caching effect is good, and of course there is still room for improvement. At the same time, we also found that if a relevant model is trained on the data of this scene for a certain scene, the cache will work better. If you have real usage scenarios and relevant data sets, welcome to communicate with us.
