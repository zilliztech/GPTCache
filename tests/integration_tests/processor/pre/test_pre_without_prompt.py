import os

from gptcache import Cache, Config
from gptcache.adapter import openai
from gptcache.manager import get_data_manager
from gptcache.processor.pre import last_content_without_prompt
from gptcache.utils.response import get_message_from_openai_answer


def test_pre_without_prompt():
    cache_obj = Cache()
    data_file = "data_map_prompt.txt"
    cache_obj.init(
        pre_embedding_func=last_content_without_prompt,
        data_manager=get_data_manager(data_path=data_file),
        config=Config(prompts=["foo"]),
    )

    if not os.path.isfile(data_file):
        cache_obj.import_data(
            [f"{i}" for i in range(10)],
            [f"receiver the foo {i}" for i in range(10)],
        )

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "foo5"},
        ],
        cache_obj=cache_obj,
    )
    assert get_message_from_openai_answer(answer) == "receiver the foo 5"
