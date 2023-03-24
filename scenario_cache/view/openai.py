import time
import openai
from ..core import cache


class ChatCompletion:
    @classmethod
    def create(cls, *args, **kwargs):
        context = kwargs.pop("cache_context", {})
        embedding_data = None
        cache_enable = cache.cache_enable_func(*args, **kwargs)
        if cache_enable:
            # start_time = time.time()
            embedding_data = cache.embedding_func(kwargs, extra_param=context.get("embedding_func", None))
            # print("embedding time: {:.2f}s".format(time.time() - start_time))

            # start_time = time.time()
            cache_data = cache.data_manager.search(embedding_data, extra_param=context.get('search', None))
            # print("search time: {:.2f}s".format(time.time() - start_time))

            rank = cache.evaluation_func(embedding_data, cache_data, extra_param=context.get('evaluation', None))
            if (cache.similarity_positive and rank >= cache.similarity_threshold) \
                    or (not cache.similarity_positive and rank <= cache.similarity_threshold):
                return_message = cache.data_manager.get_scalar_data(cache_data,
                                                                    extra_param=context.get('get_scalar_data', None))
                if return_message is not None:
                    return construct_resp_from_cache(return_message)

        # TODO support stream data
        openai_data = openai.ChatCompletion.create(*args, **kwargs)
        if cache_enable:
            cache.data_manager.save(get_message_from_openai_answer(openai_data), embedding_data,
                                    extra_param=context.get('save', None))
        return openai_data


def construct_resp_from_cache(return_message):
    return {
        "choices": [
            {
                'message': {
                    'role': 'assistant',
                    'content': return_message
                },
                'finish_reason': 'stop',
                'index': 0
            }
        ]
    }


def get_message_from_openai_answer(openai_data):
    return openai_data['choices'][0]['message']['content']
