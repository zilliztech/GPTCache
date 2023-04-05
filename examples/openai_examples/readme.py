import os
import time

import openai


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


# Before running this case, make sure the OPENAI_API_KEY environment variable is set

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

# GPT Cache exact matching usage
print("GPT Cache exact matching example.....")
print("Cache loading.....")

# To use GPT cache, that's all you need
# -------------------------------------------------
from gptcache.core import cache
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

# GPT Cache similar search usage
print("GPT Cache similar search example.....")
print("Cache loading.....")

from gptcache.core import cache
from gptcache.adapter import openai
from gptcache.embedding import Towhee
from gptcache.cache.factory import get_ss_data_manager
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation

towhee = Towhee()
data_manager = get_ss_data_manager("sqlite", "faiss", dimension=towhee.dimension)
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
