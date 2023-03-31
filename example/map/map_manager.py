import os

from gpt_cache.cache.factory import get_data_manager
from gpt_cache.view import openai
from gpt_cache.core import cache, Cache


def run():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    bak_cache = Cache()
    bak_data_file = dir_name + "/data_map_bak.txt"
    bak_cache.init(data_manager=get_data_manager("map",
                                                 data_path=bak_data_file,
                                                 max_size=10))
    data_file = dir_name + "/data_map.txt"
    cache.init(data_manager=get_data_manager("map",
                                             data_path=data_file,
                                             max_size=10),
               next_cache=bak_cache)

    cache.set_openai_key()
    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo15"}
    ]

    if not os.path.isfile(bak_data_file):
        for i in range(10):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            cache.data_manager.save(question, answer, cache.embedding_func(question))
    if not os.path.isfile(data_file):
        for i in range(10, 20):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            bak_cache.data_manager.save(question, answer, bak_cache.embedding_func(question))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)


if __name__ == '__main__':
    run()
