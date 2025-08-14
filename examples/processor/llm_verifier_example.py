import time
import os

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.post import LlmVerifier
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

print("This example demonstrates how to use LLM verification with OpenAI's GPT-3.5 Turbo model.")
cache.set_openai_key()

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})




custom_prompt = """You are a helpful assistant. Your task is to verify whether the answer is semantically consistent with the question.
If the answer is consistent, respond with "yes". If it is not consistent, respond with "no".
You must only respond in "yes" or "no". """

verifier = LlmVerifier(client=None,
                       system_prompt=custom_prompt,
                       model="gpt-3.5-turbo")

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=verifier
)

question = 'what is github'

for _ in range(3):
    start = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'user',
            'content': question
        }],
    )
    print(f"Response: {response['choices'][0]['message']['content']}")
    print(f"Time: {round(time.time() - start, 2)}s\n")
