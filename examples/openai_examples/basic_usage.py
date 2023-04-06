import os
import time

from gptcache.cache.factory import get_data_manager, get_ss_data_manager
from gptcache.core import cache, Cache, Config
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation
from gptcache.adapter import openai


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


def cache_init():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    cache.init(data_manager=get_data_manager("map"))
    os.environ["OPENAI_API_KEY"] = "API KEY"
    cache.set_openai_key()


def base_request():
    for _ in range(2):
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'user',
                    'content': 'Count to 5, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'
                }
            ],
            temperature=0,
        )
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f'Received: {response_text(response)}')


def stream_request():
    for _ in range(2):
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': "What's 1+1? Answer in one word."}
            ],
            temperature=0,
            stream=True  # this time, we set stream=True
        )

        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            collected_messages.append(chunk_message)  # save the message

        # print the time delay and text received
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        end_time = time.time()
        print("Time consuming: {:.2f}s".format(end_time - start_time))
        print(f"Full conversation received: {full_reply_content}")


def similar_request():
    onnx = Onnx()
    data_manager = get_ss_data_manager("sqlite", "faiss", dimension=onnx.dimension)
    one_cache = Cache()
    one_cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   )

    question1 = "what do you think about chatgpt"
    question2 = "what do you feel like chatgpt"

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question1}
        ],
        cache_obj=one_cache
    )
    end_time = time.time()
    print("Time consuming: {:.2f}s".format(end_time - start_time))
    print(f'Received: {response_text(answer)}')

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question2}
        ],
        cache_obj=one_cache
    )
    end_time = time.time()
    print("Time consuming: {:.2f}s".format(end_time - start_time))
    print(f'Received: {response_text(answer)}')


if __name__ == '__main__':
    cache_init()
    base_request()
    stream_request()
    similar_request()
