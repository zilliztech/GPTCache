import os
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

before = time.time()
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {'role': 'user',
     'content': 'Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.',
    }
  ],
)
print("Time Spent =", time.time() - before)
print(response["choices"])

## import gptcache
print("Cache loading.....")
from gptcache.core import cache
from gptcache.adapter import openai

## To use GPT cache, that's all you need
cache.init()
cache.set_openai_key()

before = time.time()
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {'role': 'user',
     'content': 'Summarize this for a third-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.',
    }
  ],
)
print("Time Spent =", time.time() - before)
print(response)

if "gptcache" not in response:
  response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
      {'role': 'user',
       'content': 'Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.',
       }
    ],
  )
  print("Time Spent =", time.time() - before)
  print(response)


