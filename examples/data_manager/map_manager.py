import os

from gptcache.manager import get_data_manager
from gptcache.adapter import openai
from gptcache import cache


def run():
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    data_file = dir_name + '/data_map.txt'
    data_manager = get_data_manager(data_path=data_file, max_size=10)
    cache.init(data_manager=data_manager)
    cache.set_openai_key()

    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': 'what is chatgpt'}
        ],
    )
    print(answer)


if __name__ == '__main__':
    run()
