import os
import time

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter import openai

cache.set_openai_key()

# Init cache with vector store
# if os.path.exists("faiss.index"):
#     os.remove("faiss.index")
# if os.path.exists("sqlite.db"):
#     os.remove("sqlite.db")

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )
# cache.config = Config(similarity_threshold=0.2)

question = 'what is github'

for _ in range(3):
    start = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature = 1.0,  # Change temperature here
        messages=[{
            'role': 'user',
            'content': question
        }],
    )
    print(round(time.time() - start, 3))
    print(response["choices"][0]["message"]["content"])