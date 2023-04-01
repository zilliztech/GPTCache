import os

from gptcache.cache.factory import get_data_manager
from gptcache.core import cache
from gptcache.view import openai


def run():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    data_file = dir_name + "/data_map.txt"
    cache.init(data_manager=get_data_manager("map",
                                             data_path=data_file,
                                             max_size=10))
    os.environ["OPENAI_API_KEY"] = "API KEY"
    cache.set_openai_key()
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': 'Count to 5, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
        ],
        temperature=0,
    )
    print(f'Received: {response}')

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
    print(f"Full conversation received: {full_reply_content}")


if __name__ == '__main__':
    run()
