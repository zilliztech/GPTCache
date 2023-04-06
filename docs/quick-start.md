# 😊 Quick Start

**Note**:

- You can quickly try GPTCache and put it into a production environment without heavy development. However, please note that the repository is still under heavy development.
- By default, only a limited number of libraries are installed to support the basic cache functionalities. When you need to use additional features, the related libraries will be **automatically installed**.
- Make sure that the Python version is **3.8.1 or higher**, check: `python --version`
- If you encounter issues installing a library due to a low pip version, run: `python -m pip install --upgrade pip`.

## Installation

### pip install

```bash
pip install gptcache
```

### dev install

```bash
# clone GPTCache repo
git clone https://github.com/zilliztech/GPTCache.git
cd GPTCache

# install the repo
pip install -r requirements.txt
python setup.py install
```

## Example Usage

These examples will help you understand how to use exact and similar matching in caching. 

And before running the example, **make sure** the OPENAI_API_KEY environment variable is set by executing `echo $OPENAI_API_KEY`. 

If it is not already set, it can be set by using `OPENAI_API_KEY=YOUR_API_KEY`. 

> It's important to note that this method is only effective temporarily, so if you want a permanent effect, you'll need to modify the environment variable configuration file. For instance, on a Mac, you can modify the file located at `/etc/profile`.

<details>

<summary> Click to <strong>SHOW</strong> example code </summary>

### OpenAI API original usage

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

### OpenAI API + GPTCache, exact match cache

> If you ask ChatGPT the exact same two questions, the answer to the second question will be obtained from the cache without requesting ChatGPT again.

```python
import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

print("Cache loading.....")

# To use GPTCache, that's all you need
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

### OpenAI API + GPTCache, similar search cache

> After obtaining an answer from ChatGPT in response to several similar questions, the answers to subsequent questions can be retrieved from the cache without the need to request ChatGPT again.

```python
import time


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Towhee
from gptcache.manager.factory import get_data_manager
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation

print("Cache loading.....")

towhee = Towhee()
data_manager = get_ss_data_manager("sqlite", "faiss", dimension=towhee.dimension())
cache.init(
    embedding_func=towhee.to_embeddings,
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

</details>

To use GPTCache exclusively, only the following lines of code are required, and there is no need to modify any existing code.

```python
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()
```

**More Docs**
- [System Design, how it was constructed](./system.md)
- [Features, all features currently supported by the cache](./feature.md)
- [Examples, learn better custom caching](https://github.com/zilliztech/GPTCache/blob/main/examples)