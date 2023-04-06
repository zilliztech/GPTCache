import time
from gptcache.core import cache
from gptcache.adapter import openai

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