import os
import time

from gptcache.cache.factory import get_data_manager, get_si_data_manager
from gptcache.core import cache, Cache
from gptcache.embedding import Towhee
from gptcache.similarity_evaluation.simple import pair_evaluation
from gptcache.view import openai


def run():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    data_file = dir_name + "/data_map.txt"
    cache.init(data_manager=get_data_manager("map",
                                             data_path=data_file,
                                             max_size=10))
    os.environ["OPENAI_API_KEY"] = "API KEY"
    cache.set_openai_key()

    # base request test
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': 'Count to 5, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
        ],
        temperature=0,
    )
    print(f'Received: {response}')

    # stream request test
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
        print("time consuming: {:.2f}s".format(end_time - start_time))
        print(f"Full conversation received: {full_reply_content}")

    # similarity test
    towhee = Towhee()
    data_manager = get_si_data_manager("sqlite", "faiss",
                                       dimension=towhee.dimension(), max_size=2000)
    one_cache = Cache()
    one_cache.init(embedding_func=towhee.to_embeddings,
                   data_manager=data_manager,
                   evaluation_func=pair_evaluation,
                   similarity_threshold=1,
                   similarity_positive=False)

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
    print("time consuming: {:.2f}s".format(end_time - start_time))
    print(answer)

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question2}
        ],
        cache_obj=one_cache
    )
    end_time = time.time()
    print("time consuming: {:.2f}s".format(end_time - start_time))
    print(answer)

    one_cache.close()


if __name__ == '__main__':
    run()
