import openai
from ..core import cache, time_cal


class ChatCompletion:
    @classmethod
    def create(cls, *args, **kwargs):
        chat_cache = kwargs.pop("cache_obj", cache)
        cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
        context = kwargs.pop("cache_context", {})
        embedding_data = None
        # you want to retry to send the request to chatgpt when the cache is negative
        cache_skip = kwargs.pop("cache_skip", False)
        pre_embedding_data = chat_cache.pre_embedding_func(kwargs,
                                                           extra_param=context.get("pre_embedding_func", None))
        if cache_enable and not cache_skip:
            embedding_data = time_cal(chat_cache.embedding_func,
                                      func_name="embedding",
                                      report_func=chat_cache.report.embedding,
                                      )(pre_embedding_data, extra_param=context.get("embedding_func", None))
            cache_data_list = time_cal(chat_cache.data_manager.search,
                                       func_name="search",
                                       report_func=chat_cache.report.search,
                                       )(embedding_data, extra_param=context.get('search', None))
            cache_answers = []
            for cache_data in cache_data_list:
                cache_question, cache_answer = chat_cache.data_manager.get_scalar_data(
                    cache_data, extra_param=context.get('get_scalar_data', None))
                rank = chat_cache.evaluation_func({
                    "question": pre_embedding_data,
                    "embedding": embedding_data,
                }, {
                    "question": cache_question,
                    "answer": cache_answer,
                    "search_result": cache_data,
                }, extra_param=context.get('evaluation', None))
                if (chat_cache.similarity_positive and rank >= chat_cache.similarity_threshold) \
                        or (not chat_cache.similarity_positive and rank <= chat_cache.similarity_threshold):
                    cache_answers.append((rank, cache_answer))
            cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
            if len(cache_answers) != 0:
                return_message = chat_cache.post_process_messages_func([t[1] for t in cache_answers])
                chat_cache.report.hint_cache()
                return construct_resp_from_cache(return_message)

        # TODO cache poster -> can chain
        # TODO support stream data
        openai_data = openai.ChatCompletion.create(*args, **kwargs)
        if cache_enable:
            chat_cache.data_manager.save(pre_embedding_data,
                                         get_message_from_openai_answer(openai_data),
                                         embedding_data,
                                         extra_param=context.get('save', None))
        return openai_data


def construct_resp_from_cache(return_message):
    return {
        "gpt_cache": True,
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
