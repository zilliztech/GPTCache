# Release Note

To read the following content, you need to understand the basic use of GPTCache, references:

- [Readme doc](https://github.com/zilliztech/GPTCache)
- [Usage doc](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md)

## v0.1.36 (2023.7.14)

1. Fix the connection error of the remote redis cache store
2. Add the openai proxy for the chat complete api

## v0.1.35 (2023.7.7)

1. Support the redis as the cache store, usage example: [redis+onnx](https://github.com/zilliztech/GPTCache/blob/main/tests/integration_tests/test_redis_onnx.py)
2. Add report table for easy analysis of cache data

## v0.1.34 (2023.6.30)

1. Add support for Qdrant Vector Store
2. Add support for Mongodb Cache Store
3. Fix bug about the redis vector and onnx similarity evaluation

## v0.1.33 (2023.6.27)

1. Fix the eviction error
2. Add a flag for search only operation
3. Support to change the redis namespace
4. Add `How to better configure your cache` document

## v0.1.32 (2023.6.15)

1. Support the redis as vector store

```python
from gptcache.manager import VectorBase

vector_base = VectorBase("redis", dimension=10)
```

2. Fix the context len config bug

## v0.1.31 (2023.6.14)

1. To improve the precision of cache hits, two similarity evaluation methods were added

a. SBERT CrossEncoder Evaluation

```python
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
evaluation = SbertCrossencoderEvaluation()
score = evaluation.evaluation(
    {
        'question': 'What is the color of sky?'
    },
    {
        'question': 'hello'
    }
)
```

b. Cohere rerank api (**Free accounts can make up to 100 calls per minute.**)

```python
from gptcache.similarity_evaluation import CohereRerankEvaluation

evaluation = CohereRerankEvaluation()
score = evaluation.evaluation(
    {
        'question': 'What is the color of sky?'
    },
    {
        'answer': 'the color of sky is blue'
    }
)
```

c. Multi-round dialog similarity weight matching

```python
from gptcache.similarity_evaluation import SequenceMatchEvaluation

weights = [0.5, 0.3, 0.2]
evaluation = SequenceMatchEvaluation(weights, 'onnx')

query = {
    'question': 'USER: "foo2" USER: "foo4"',
}

cache = {
    'question': 'USER: "foo6" USER: "foo8"',
}

score = evaluation.evaluation(query, cache)
```

d. Time Evaluation. For the cached answer, first check the time dimension, such as only using the generated cache for the past day

```python
from gptcache.similarity_evaluation import TimeEvaluation

evaluation = TimeEvaluation(evaluation="distance", time_range=86400)

similarity = eval.evaluation(
    {},
    {
        "search_result": (3.5, None),
        "cache_data": CacheData("a", "b", create_on=datetime.datetime.now()),
    },
)
```

2. Fix some bugs

a. OpenAI exceptions type #416
b. LangChainChat does work with _agenerate function #400

## v0.1.30 (2023.6.7)

1. Support to use the cohere rerank api to evaluate the similarity

```python
from gptcache.similarity_evaluation import CohereRerankEvaluation

evaluation = CohereRerankEvaluation()
score = evaluation.evaluation(
    {
        'question': 'What is the color of sky?'
    },
    {
        'answer': 'the color of sky is blue'
    }
)
```

2. Improve the gptcache server api, refer to the "/docs" path after starting the server
3. Fix the bug about the langchain track token usage

## v0.1.29 (2023.6.2)

1. Improve the GPTCache server by using FASTAPI

**NOTE**: The api struct has been optimized, details: [Use GPTCache server](https://github.com/zilliztech/GPTCache/blob/dev/docs/usage.md#use-gptcache-server)

2. Add the usearch vector store

```python
from gptcache.manager import manager_factory

data_manager = manager_factory("sqlite,usearch", vector_params={"dimension": 10})
```

## v0.1.28 (2023.5.29)
To handle a large prompt, there are currently two options available:

1. Increase the column size of CacheStorage.

```python
from gptcache.manager import manager_factory

data_manager = manager_factory(
    "sqlite,faiss", scalar_params={"table_len_config": {"question_question": 5000}}
)

```
More Details:
- 'question_question': the question column size in the question table, default to 3000.
- 'answer_answer': the answer column size in the answer table, default to 3000.
- 'session_id': the session id column size in the session table, default to 1000.
- 'dep_name': the name column size in the dep table, default to 1000.
- 'dep_data': the data column size in the dep table, default to 3000.

2. When using a template, use the dynamic value in the template as the cache key instead of using the entire template as the key.

- **str template**
```python
from gptcache import Config
from gptcache.processor.pre import last_content_without_template

template_obj = "tell me a joke about {subject}"
prompt = template_obj.format(subject="animal")
value = last_content_without_template(
    data={"messages": [{"content": prompt}]}, cache_config=Config(template=template_obj)
)
print(value)
# ['animal']
```

- **langchain prompt template**

```python
from langchain import PromptTemplate

from gptcache import Config
from gptcache.processor.pre import last_content_without_template

template_obj = PromptTemplate.from_template("tell me a joke about {subject}")
prompt = template_obj.format(subject="animal")

value = last_content_without_template(
    data={"messages": [{"content": prompt}]},
    cache_config=Config(template=template_obj.template),
)
print(value)
# ['animal']
```

3. Wrap the openai object, reference: [BaseCacheLLM](https://gptcache.readthedocs.io/en/dev/references/adapter.html#module-gptcache.adapter.base)

```python
import random

from gptcache import Cache
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import last_content

cache_obj = Cache()
init_similar_cache(
    data_dir=str(random.random()), pre_func=last_content, cache_obj=cache_obj
)


def proxy_openai_chat_complete(*args, **kwargs):
    nonlocal is_proxy
    is_proxy = True
    import openai as real_openai

    return real_openai.ChatCompletion.create(*args, **kwargs)


openai.ChatCompletion.llm = proxy_openai_chat_complete
openai.ChatCompletion.cache_args = {"cache_obj": cache_obj}

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's GitHub"},
    ],
)
```

## v0.1.27 (2023.5.25)
1. Support the uform embedding, which can be used the **bilingual** (english + chinese) language

```python
from gptcache.embedding import UForm

test_sentence = 'Hello, world.'
encoder = UForm(model='unum-cloud/uform-vl-english')
embed = encoder.to_embeddings(test_sentence)

test_sentence = '什么是Github'
encoder = UForm(model='unum-cloud/uform-vl-multilingual')
embed = encoder.to_embeddings(test_sentence)
```

## v0.1.26 (2023.5.23)

1. Support the paddlenlp embedding

```python
from gptcache.embedding import PaddleNLP

test_sentence = 'Hello, world.'
encoder = PaddleNLP(model='ernie-3.0-medium-zh')
embed = encoder.to_embeddings(test_sentence)
```

2. Support [the openai Moderation api](https://platform.openai.com/docs/api-reference/moderations)

```python
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import get_openai_moderation_input

init_similar_cache(pre_func=get_openai_moderation_input)
openai.Moderation.create(
    input="hello, world",
)
```

3. Add the llama_index bootcamp, through which you can learn how GPTCache works with llama index

details: [WebPage QA](https://gptcache.readthedocs.io/en/latest/bootcamp/llama_index/webpage_qa.html)

## v0.1.25 (2023.5.18)

1. Support the DocArray vector database

```python
from gptcache.manager import manager_factory

data_manager = manager_factory("sqlite,docarray")
```

2. Add rwkv model for embedding

```python
from gptcache.embedding import Rwkv

test_sentence = 'Hello, world.'
encoder = Rwkv(model='sgugger/rwkv-430M-pile')
embed = encoder.to_embeddings(test_sentence)
```

## v0.1.24 (2023.5.15)

1. Support the langchain embedding

```python
from gptcache.embedding import LangChain
from langchain.embeddings.openai import OpenAIEmbeddings

test_sentence = 'Hello, world.'
embeddings = OpenAIEmbeddings(model="your-embeddings-deployment-name")
encoder = LangChain(embeddings=embeddings)
embed = encoder.to_embeddings(test_sentence)
```

2. Add gptcache client

```python
from gptcache import Client

client = Client()
client.put("Hi", "Hi back")
ans = client.get("Hi")
```

3. Support pgvector as vector store

```python
from gptcache.manager import manager_factory

data_manager = manager_factory("sqlite,pgvector", vector_params={"dimension": 10})
```

4. Add the GPTCache server doc

reference: https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md#Build-GPTCache-server

## v0.1.23 (2023.5.11)

1. Support the session for the `LangChainLLMs`

```python
from langchain import OpenAI
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.session import Session

session = Session(name="sqlite-example")
llm = LangChainLLMs(llm=OpenAI(temperature=0), session=session)
```

2. Optimize the summarization context process

```python
from gptcache import cache
from gptcache.processor.context.summarization_context import SummarizationContextProcess

context_process = SummarizationContextProcess()
cache.init(
    pre_embedding_func=context_process.pre_process,
)
```

3. Add BabyAGI bootcamp

details: https://github.com/zilliztech/GPTCache/blob/main/docs/bootcamp/langchain/baby_agi.ipynb

## v0.1.22 (2023.5.7)

1. Process the dialog context through the context processing interface, which currently supports two ways: summarize and selective context

```python
import transformers
from gptcache.processor.context.summarization_context import SummarizationContextProcess
from gptcache.processor.context.selective_context import SelectiveContextProcess
from gptcache import cache

summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")
context_process = SummarizationContextProcess(summarizer, None, 512)
cache.init(
    pre_embedding_func=context_process.pre_process,
)

context_processor = SelectiveContextProcess()
cache.init(
    pre_embedding_func=context_process.pre_process,
)
```

## v0.1.21 (2023.4.29)

1. Support the temperature param

```python
from gptcache.adapter import openai

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 1.0,  # Change temperature here
    messages=[{
        "role": "user",
        "content": question
    }],
)
```

2. Add the session layer

```python
from gptcache.adapter import openai
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

details: https://github.com/zilliztech/GPTCache/tree/main/examples#How-to-run-with-session

3. Support config cache with yaml for server

```python
from gptcache.adapter.api import init_similar_cache_from_config

init_similar_cache_from_config(config_dir="cache_config_template.yml")
```

config file template: https://github.com/zilliztech/GPTCache/blob/main/cache_config_template.yml

4. Adapt the dolly model

```python
from gptcache.adapter.dolly import Dolly

llm = Dolly.from_model(model="databricks/dolly-v2-3b")
llm(question)
```

## v0.1.20 (2023.4.26)

1. support the `temperature` param, like openai

A non-negative number of sampling temperature, defaults to 0.
A higher temperature makes the output more random.
A lower temperature means a more deterministic and confident output.

2. Add llama adapter

```python
from gptcache.adapter.llama_cpp import Llama

llm = Llama('./models/7B/ggml-model.bin')
answer = llm(prompt=question)
```

## v0.1.19 (2023.4.24)

1. Add stability sdk adapter (text -> image)

```python
import os
import time

from gptcache import cache
from gptcache.processor.pre import get_prompt
from gptcache.adapter.stability_sdk import StabilityInference, generation
from gptcache.embedding import Onnx
from gptcache.manager.factory import manager_factory
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# init gptcache
onnx = Onnx()
data_manager = manager_factory('sqlite,faiss,local', 
                               data_dir='./', 
                               vector_params={'dimension': onnx.dimension},
                               object_params={'path': './images'}
                               )
cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
    )

api_key = os.getenv('STABILITY_KEY', 'key-goes-here')

stability_api = StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=False, # Print debug messages.
    engine='stable-diffusion-xl-beta-v2-2-2', # Set the engine to use for generation.
)

start = time.time()
answers = stability_api.generate(
    prompt='a cat sitting besides a dog',
    width=256,
    height=256
    )
```

stability reference: https://platform.stability.ai/docs/features/text-to-image

2. Add minigpt4 adapter

Notice: It cannot be used directly, it needs to cooperate with mini-GPT4 source code, refer to: https://github.com/Vision-CAIR/MiniGPT-4/pull/136

## v0.1.18 (2023.4.23)

1. Add vqa bootcamp

reference: https://github.com/zilliztech/GPTCache/tree/main/docs/bootcamp/replicate

2. Add two streamlit multimodal demos

reference: https://github.com/zilliztech/GPTCache/tree/main/docs/bootcamp/streamlit

3. Add vit image embedding func

```python
from gptcache.embedding import ViT

encoder = ViT(model="google/vit-base-patch16-384")
embed = encoder.to_embeddings(image)
```

4. Add `init_similar_cache` func for the GPTCache api module

```python
from gptcache.adapter.api import init_similar_cache

init_similar_cache("cache_data")
```

5. The simple GPTCache server provides similar cache

- clone the GPTCache repo, `git clone https://github.com/zilliztech/GPTCache.git`
- install the gptcache model, `pip install gptcache`
- run the GPTCache server, `cd gptcache_server && python server.py`

## v0.1.17 (2023.4.20)

1. Add image embedding timm

```python
import requests
from PIL import Image
from gptcache.embedding import Timm

url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
image = Image.open(requests.get(url, stream=True).raw)  # Read image url as PIL.Image      
encoder = Timm(model='resnet18')
image_tensor = encoder.preprocess(image)
embed = encoder.to_embeddings(image_tensor)
```

2. Add Replicate adapter, vqa (visual question answering) (**experimental**)

```python
from gptcache.adapter import replicate

question = "what is in the image?"

replicate.run(
    "andreasjansson/blip-2:xxx",
    input={
        "image": open(image_path, 'rb'),
        "question": question
    }
)
```

3. Support to flush data for preventing accidental loss of memory data

```python
from gptcache import cache

cache.flush()
```

## v0.1.16 (2023.4.18)

1. Add StableDiffusion adapter (**experimental**)

```python
import torch

from gptcache.adapter.diffusers import StableDiffusionPipeline
from gptcache.processor.pre import get_prompt
from gptcache import cache

cache.init(
    pre_embedding_func=get_prompt,
)
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

prompt = "a photo of an astronaut riding a horse on mars"
pipe(prompt=prompt).images[0]
```

2. Add speech to text bootcamp, [link](https://github.com/zilliztech/GPTCache/tree/main/docs/bootcamp/openai/speech_to_text.ipynb)

3. More convenient management of cache files

```python
from gptcache.manager.factory import manager_factory

data_manager = manager_factory('sqlite,faiss', data_dir="test_cache", vector_params={"dimension": 5})
```

4. Add a simple GPTCache server (**experimental**)

After starting this server, you can:

- put the data to cache, like: `curl -X PUT -d "receive a hello message" "http://localhost:8000?prompt=hello"`
- get the data from cache, like: `curl -X GET  "http://localhost:8000?prompt=hello"`

Currently the service is just a map cache, more functions are still under development.

## v0.1.15 (2023.4.17)

1. Add GPTCache api, makes it easier to access other different llm models and applications

```python
from gptcache.adapter.api import put, get
from gptcache.processor.pre import get_prompt
from gptcache import cache

cache.init(pre_embedding_func=get_prompt)
put("hello", "foo")
print(get("hello"))
```

2. Add image generation bootcamp, link: https://github.com/zilliztech/GPTCache/blob/main/docs/bootcamp/openai/image_generation.ipynb

## v0.1.14 (2023.4.17)

1. Fix to fail to save the data to cache

## ~~v0.1.13 (2023.4.16)~~ Don't Use it, should use the `v0.1.14`

1. Add openai audio adapter (**experimental**)

```python
cache.init(pre_embedding_func=get_file_bytes)

openai.Audio.transcribe(
    model="whisper-1",
    file=audio_file
)
```

2. Improve data eviction implementation

In the future, users will have greater flexibility to customize eviction methods, such as by using Redis or Memcached. Currently, the default caching library is cachetools, which provides an in-memory cache. Other libraries are not currently supported, but may be added in the future.

## v0.1.12 (2023.4.15)

1. The llm request can customize topk search parameters

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question},
    ],
    top_k=10,
)
```

## v0.1.11 (2023.4.14)

1. Add openai complete adapter

```python
cache.init(pre_embedding_func=get_prompt)

response = openai.Completion.create(
                model="text-davinci-003",
                prompt=question
            )
```

2. Add langchain and openai [bootcamp](https://github.com/zilliztech/GPTCache/tree/main/docs/bootcamp)

3. Add openai image adapter (**experimental**)

```python
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()

prompt1 = 'a cat sitting besides a dog'
size1 = '256x256'

openai.Image.create(
    prompt=prompt1,
    size=size1,
    response_format='b64_json'
    )
```

4. Refine storage interface

## v0.1.10 (2023.4.13)

1. Add kreciprocal similarity evaluation

K-reprciprocl evaluation is a method inspired by the popular reranking method in ReID(https://arxiv.org/abs/1701.08398). The term “k-reciprocal” comes from the fact that the algorithm creates reciprocal relationships between similar embeddings in the top-k list. In other words, if embedding A is similar to embedding B and embedding B is similar to embedding A, then A and B are said to be “reciprocally similar” to each other. This evaluation abandon those embeddings pairs which are not “reciprocally similar” in their K nearest neighbors. And the remaining pairs will keep the distance for the final rank.

```python
vector_base = VectorBase("faiss", dimension=d)
data_manager = get_data_manager(CacheBase("sqlite"), vector_base)
evaluation = KReciprocalEvaluation(vectordb=vector_base)
cache.init(
    ... # other configs
    data_manager=data_manager,
    similarity_evaluation=evaluation,
)
```

2. Add LangChainChat adapter

```python
from gptcache.adapter.langchain_models import LangChainChat

cache.init(
    pre_embedding_func=get_msg,
)

chat = LangChainChat(chat=ChatOpenAI(temperature=0))
answer = chat(
    messages=[
        HumanMessage(
            content="Translate this sentence from English to Chinese. I love programming."
        )
    ]
)
```

## v0.1.9 (2023.4.12)

1. Import data into cache

```python
cache.init()

questions = ["foo1", "foo2"]
answers = ["a1", "a2"]
cache.import_data(questions=questions, answers=answers)
```

2. New pre-process function: remove prompts

When using the LLM model, a prompt may be added for each input. If the entire message with the prompt is brought into the cache, it may lead to an increase in the cache error hit rate. For example, the text of the prompt is very long, and the text of the real question is very short. .

```python
cache_obj.init(
    pre_embedding_func=last_content_without_prompt,
    config=Config(prompts=["foo"]),
)
```

3. Embedded milvus

The embedded Milvus is a lightweight version of Milvus that can be embedded into your Python application. It is a single binary that can be easily installed and run on your machine.

```python
with TemporaryDirectory(dir="./") as root:
    db = VectorBase(
                    "milvus",
                    local_mode=True,
                    local_data=str(root),
                    ... #other config
                )
    data_manager = get_data_manager("sqlite", vector_base)
```
