import os

import gpt_cache.core
from gpt_cache.cache.data_manager import MapDataManager
from gpt_cache.view import openai
from gpt_cache.core import cache


def run():
    dirname, _ = os.path.split(os.path.abspath(__file__))
    cache.init(data_manager=MapDataManager(dirname + "/data_map.txt"))
    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ]

    # you should OPEN it if you FIRST run it
    # cache.data_manager.save("receiver the foo", cache.embedding_func({"messages": mock_messages}))
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
        cache_context={
            "search": {
                "user": "foo"
            }
        },
    )
    print(answer)
    cache.data_manager.close()


if __name__ == '__main__':
    run()
