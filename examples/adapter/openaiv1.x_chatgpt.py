import time

from gptcache import cache
from gptcache.adapter import openai
from openai import OpenAI

cache.init()

client = OpenAI()

def openai_chat():
  question = 'what is github'
  start = time.time()
  answer = openai.cache_openai_chat_complete(
        client,
        model='gpt-3.5-turbo',
        messages=[
          {
              'role': 'user',
              'content': question
          }
        ],
      )
  print(answer)
  print(time.time() - start)

openai_chat()
openai_chat()
