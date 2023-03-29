import os

from gpt_cache.cache.factory import get_data_manager
from gpt_cache.view import openai
from gpt_cache.core import cache


def run():
    dirname, _ = os.path.split(os.path.abspath(__file__))
    cache.init(data_manager=get_data_manager("map",
                                             data_path=dirname + "/data_map.txt",
                                             max_size=10))
    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo5"}
    ]

    # you should CLOSE it if you SECONDLY run it
    for i in range(10):
        question = f"foo{i}"
        answer = f"receiver the foo {i}"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)


if __name__ == '__main__':
    run()
