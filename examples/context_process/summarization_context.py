import os
import time

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.context import SummarizationContextProcess
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


def response_text(openai_resp):
    return openai_resp["choices"][0]["message"]["content"]


def cache_init():
    onnx = Onnx()
    context_process = SummarizationContextProcess()
    data_manager = manager_factory(
        "sqlite,faiss", vector_params={"dimension": onnx.dimension}
    )
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    cache.init(
        pre_embedding_func=context_process.pre_process,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    os.environ["OPENAI_API_KEY"] = "API KEY"
    cache.set_openai_key()


def base_request():
    cache_init()
    for _ in range(2):
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Can you give me some tips for staying focused while working from home?",
                },
                {
                    "role": "system",
                    "content": "Sure! Here are some tips: create a designated workspace, set a schedule, take breaks, minimize distractions, and practice good time management.",
                },
                {
                    "role": "user",
                    "content": "Those are all great suggestions. Do you have any tips for maintaining a healthy work-life balance while working from home?",
                },
                {
                    "role": "system",
                    "content": "Definitely! Setting clear boundaries between work and personal time, scheduling regular breaks throughout the day, and finding ways to disconnect from work after hours can help. Additionally, make time for hobbies and other activities you enjoy outside of work to help you relax and recharge.",
                },
                {"role": "user", "content": "can you give meore tips?"},
            ],
            temperature=0,
        )
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Received: {response_text(response)}")


if __name__ == "__main__":
    base_request()
