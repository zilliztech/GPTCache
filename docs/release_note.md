# 🎉 Introduction to new functions of GPTCache

To read the following content, you need to understand the basic use of GPTCache, references:

- [Readme doc](https://github.com/zilliztech/GPTCache)
- [Usage doc](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md)

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
