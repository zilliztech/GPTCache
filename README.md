# GPTCache : A Library for Creating Semantic Cache for LLM Queries
Slash Your LLM API Costs by 10x 💰, Boost Speed by 100x ⚡ 

[![Release](https://img.shields.io/pypi/v/gptcache?label=Release&color)](https://pypi.org/project/gptcache/)
[![CI](https://github.com/zilliztech/gptcache/actions/workflows/CI_main.yaml/badge.svg)](https://github.com/zilliztech/gptcache/actions/workflows/CI_main.yaml)
[![pip download](https://img.shields.io/pypi/dm/gptcache.svg?color=bright-green)](https://pypi.org/project/gptcache/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/zilliz_universe.svg?style=social&label=Follow%20%40Zilliz)](https://twitter.com/zilliz_universe)
[![Discord](https://dcbadge.vercel.app/api/server/Q8C6WEjSWV?compact=true&style=flat)](https://discord.gg/Q8C6WEjSWV)

Documentation: https://gptcache.readthedocs.io/en/latest/.

Discord: https://discord.gg/Q8C6WEjSWV.

Twitter: https://twitter.com/zilliz_universe.


## Quick Install

`pip install gptcache`

## 🚀 What is GPTCache?

ChatGPT and various large language models (LLMs) boast incredible versatility, enabling the development of a wide range of applications. However, as your application grows in popularity and encounters higher traffic levels, the expenses related to LLM API calls can become substantial. Additionally, LLM services might exhibit slow response times, especially when dealing with a significant number of requests.

To tackle this challenge, we have created GPTCache, a project dedicated to building a semantic cache for storing LLM responses. This library offers the following primary benefits:

1. **Decreased expenses**: Most LLM services charge fees based on a combination of number of requests and [token count](https://openai.com/pricing). By caching query results, GPTCache reduces both the number of requests and the number of tokens sent to the LLM service. This minimizes the overall cost of using the service. 
2. **Enhanced performance**: LLMs employ generative AI algorithms to generate responses in real-time, a process that can sometimes be time-consuming. However, when a similar query is cached, the response time significantly improves, as the result is fetched directly from the cache, eliminating the need to interact with the LLM service. In most situations, GPTCache can also provide superior query throughput compared to standard LLM services.
3. **Improved scalability and availability**: LLM services frequently enforce [rate limits](https://platform.openai.com/docs/guides/rate-limits), which are constraints that APIs place on the number of times a user or client can access the server within a given timeframe. Hitting a rate limit means that additional requests will be blocked until a certain period has elapsed, leading to a service outage. With GPTCache, you can easily scale to accommodate an increasing volume of of queries, ensuring consistent performance as your application's user base expands.
4. **Flexible development environment**: When developing LLM applications, an LLM APIs connection is required to prove concepts. GPTCache offers the same interface as LLM APIs and can store LLM-generated or mocked data. This helps to verify your application's features without connecting to the LLM APIs or even the network.

## 🤔 How does it work?

A good analogy for GptCache is to think of it as a more semantic version of Redis. In GPTCache, hits are not limited to exact matches, but rather also include prompts and context similar to previous queries. We believe that the traditional cache design still works for AIGC applications due to the following reasons:

- Locality is present everywhere. Like traditional application systems, AIGC applications also face similar hot topics. For instance, ChatGPT itself may be a popular topic among programmers.
- For purpose-built SaaS services, users tend to ask questions within a specific domain, with both temporal and spatial locality.
- By utilizing vector similarity search, it is possible to find a similarity relationship between questions and answers at a relatively low cost.

We provide [benchmarks](https://github.com/zilliztech/gpt-cache/blob/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py) to illustrate the concept. In semantic caching, there are three key measurement dimensions: false positives, false negatives, and hit latency. With the plugin-style implementation, users can easily tradeoff these three measurements according to their needs.

## 😊 Quick Start

**Note**:

- You can quickly try GPTCache and put it into a production environment without heavy development. However, please note that the repository is still under heavy development.
- By default, only a limited number of libraries are installed to support the basic cache functionalities. When you need to use additional features, the related libraries will be **automatically installed**.
- Make sure that the Python version is **3.8.1 or higher**, check: `python --version`
- If you encounter issues installing a library due to a low pip version, run: `python -m pip install --upgrade pip`.

### dev install

```bash
# clone GPTCache repo
git clone https://github.com/zilliztech/GPTCache.git
cd GPTCache

# install the repo
pip install -r requirements.txt
python setup.py install
```

### example usage

These examples will help you understand how to use exact and similar matching with caching. 

Before running the example, **make sure** the OPENAI_API_KEY environment variable is set by executing `echo $OPENAI_API_KEY`. 

If it is not already set, it can be set by using `export OPENAI_API_KEY=YOUR_API_KEY` on Unix/Linux/MacOS systems or `set OPENAI_API_KEY=YOUR_API_KEY` on Windows systems. 

> It is important to note that this method is only effective temporarily, so if you want a permanent effect, you'll need to modify the environment variable configuration file. For instance, on a Mac, you can modify the file located at `/etc/profile`.

<details>

<summary> Click to <strong>SHOW</strong> example code </summary>

#### OpenAI API original usage

```python
import os
import time

import openai


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


question = 'what‘s chatgpt'

# OpenAI API original usage
openai.api_key = os.getenv("OPENAI_API_KEY")
start_time = time.time()
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {
        'role': 'user',
        'content': question
    }
  ],
)
print(f'Question: {question}')
print("Time consuming: {:.2f}s".format(time.time() - start_time))
print(f'Answer: {response_text(response)}\n')

```

#### OpenAI API + GPTCache, exact match cache

> If you ask ChatGPT the exact same two questions, the answer to the second question will be obtained from the cache without requesting ChatGPT again.

```python
import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

print("Cache loading.....")

# To use GPTcache, that's all you need
# -------------------------------------------------
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()
# -------------------------------------------------

question = "what's github"
for _ in range(2):
    start_time = time.time()
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response_text(response)}\n')
```

#### OpenAI API + GPTCache, similar search cache

> After obtaining an answer from ChatGPT in response to several similar questions, the answers to subsequent questions can be retrieved from the cache without the need to request ChatGPT again.

```python
import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

print("Cache loading.....")

onnx = Onnx()
data_manager = get_data_manager("sqlite", "faiss", dimension=onnx.dimension)
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

questions = [
    "what's github",
    "can you explain what GitHub is",
    "can you tell me more about GitHub"
    "what is the purpose of GitHub"
]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response_text(response)}\n')
```

</details>

To use GPTCache exclusively, only the following lines of code are required, and there is no need to modify any existing code.

```python
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()
```

More Docs：

- [Features, all features currently supported by the cache](docs/feature.md)
- [Examples, learn better custom caching](examples/README.md)

## 🤗 Modules

![GPTCache Struct](docs/GPTCacheStructure.png)

- **LLM Adapter**: 
The LLM Adapter is designed to integrate different LLM models by unifying their APIs and request protocols. GPTCache offers a standardized interface for this purpose, with current support for ChatGPT integration.
  - [x] Support OpenAI ChatGPT API.
  - [x] Support langchain.
  - [ ] Support other LLMs, such as Hugging Face Hub, Bard, Anthropic, and self-hosted models like LLaMa.
- **Embedding Generator**: 
This module is created to extract embeddings from requests for similarity search. GPTCache offers a generic interface that supports multiple embedding APIs, and presents a range of solutions to choose from. 
  - [x] Disable embedding. This will turn GPTCache into a keyword-matching cache.
  - [x] Support OpenAI embedding API.
  - [x] Support [ONNX](https://onnx.ai/) with the GPTCache/paraphrase-albert-onnx model.
  - [x] Support [Hugging Face](https://huggingface.co/) embedding API.
  - [x] Support [Cohere](https://docs.cohere.ai/reference/embed) embedding API.
  - [x] Support [fastText](https://fasttext.cc) embedding API.
  - [x] Support [SentenceTransformers](https://www.sbert.net) embedding API.
  - [ ] Support other embedding apis
- **Cache Storage**:
**Cache Storage** is where the response from LLMs, such as ChatGPT, is stored. Cached responses are retrieved to assist in evaluating similarity and are returned to the requester if there is a good semantic match. At present, GPTCache supports SQLite and offers a universally accessible interface for extension of this module.
  - [x] Support [SQLite](https://sqlite.org/docs.html).
  - [x] Support [PostgreSQL](https://www.postgresql.org/).
  - [x] Support [MySQL](https://www.mysql.com/).
  - [x] Support [MariaDB](https://mariadb.org/).
  - [x] Support [SQL Server](https://www.microsoft.com/en-us/sql-server/).
  - [x] Support [Oracle](https://www.oracle.com/).
  - [ ] Support [MongoDB](https://www.mongodb.com/).
  - [ ] Support [Redis](https://redis.io/).
  - [ ] Support [Minio](https://min.io/).
  - [ ] Support [HBase](https://hbase.apache.org/).
  - [ ] Support [ElasticSearch](https://www.elastic.co/)
  - [ ] Support [zincsearch](https://zinc.dev/)
  - [ ] Support other storages
- **Vector Store**:
The **Vector Store** module helps find the K most similar requests from the input request's extracted embedding. The results can help assess similarity. GPTCache provides a user-friendly interface that supports various vector stores, including Milvus, Zilliz Cloud, and FAISS. More options will be available in the future.
  - [x] Support [Milvus](https://milvus.io/).
  - [x] Support [Zilliz Cloud](https://cloud.zilliz.com/).
  - [x] Support [FAISS](https://faiss.ai/).
  - [ ] Support other vector databases
- **Cache Manager**:
The **Cache Manager** is responsible for controlling the operation of both the **Cache Storage** and **Vector Store**.
  - **Eviction Policy**:
  Currently, GPTCache makes decisions about evictions based solely on the number of lines. This approach can result in inaccurate resource evaluation and may cause out-of-memory (OOM) errors. We are actively investigating and developing a more sophisticated strategy.
    - [x] LRU eviction policy
    - [x] FIFO eviction policy
    - [ ] More complicated eviction policies
- **Similarity Evaluator**: 
This module collects data from both the **Cache Storage** and **Vector Store**, and uses various strategies to determine the similarity between the input request and the requests from the **Vector Store**. Based on this similarity, it determines whether a request matches the cache. GPTCache provides a standardized interface for integrating various strategies, along with a collection of implementations to use. The following similarity definitions are currently supported or will be supported in the future:
  - [x] The distance we obtain from the **Vector Store**.
  - [x] A model-based similarity determined using the GPTCache/albert-duplicate-onnx model from [ONNX](https://onnx.ai/).
  - [x] Exact matches between the input request and the requests obtained from the **Vector Store**.
  - [x] Distance represented by applying linalg.norm from numpy to the embeddings.
  - [ ] BM25 and other similarity measurements
  - [ ] Support other model serving framework such as PyTorch
 
  
  **Note**:Not all combinations of different modules may be compatible with each other. For instance, if we disable the **Embedding Extractor**, the **Vector Store** may not function as intended. We are currently working on implementing a combination sanity check for **GPTCache**.

## 😇 Roadmap
Coming soon! [Stay tuned!](https://twitter.com/zilliz_universe)

## 😍 Contributing
We are extremely open to contributions, be it through new features, enhanced infrastructure, or improved documentation.

For comprehensive instructions on how to contribute, please refer to our [contribution guide](docs/contributing.md).
