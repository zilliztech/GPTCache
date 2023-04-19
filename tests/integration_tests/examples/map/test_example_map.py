import os
from tempfile import TemporaryDirectory

from gptcache.manager.factory import get_data_manager
from gptcache.adapter import openai
from gptcache import cache, Cache


def test_map():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    bak_cache = Cache()
    bak_data_file = dir_name + "/data_map_bak.txt"
    bak_cache.init(data_manager=get_data_manager(data_path=bak_data_file, max_size=10))
    data_file = dir_name + "/data_map.txt"
    cache.init(
        data_manager=get_data_manager(data_path=data_file, max_size=10),
        next_cache=bak_cache,
    )

    cache.set_openai_key()
    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo15"},
    ]

    if not os.path.isfile(bak_data_file):
        cache.import_data(
            [f"foo{i}" for i in range(10)], [f"receiver the foo {i}" for i in range(10)]
        )
    if not os.path.isfile(data_file):
        bak_cache.import_data(
            [f"foo{i}" for i in range(10, 20)],
            [f"receiver the foo {i}" for i in range(10, 20)],
        )

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)
