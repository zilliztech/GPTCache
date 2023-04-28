import logging
from gptcache import cache
from gptcache.processor.post import temperature_softmax
from gptcache.utils.error import NotInitError
from gptcache.utils.time import time_cal


def adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs):
    user_temperature = "temperature" in kwargs
    user_top_k = "top_k" in kwargs
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache)
    session = kwargs.pop("session", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    # you want to retry to send the request to chatgpt when the cache is negative
    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        cache_skip = kwargs.pop("cache_skip", temperature_softmax(
            messages=cache_skip_options, scores = prob_cache_skip, temperature=temperature))
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = chat_cache.pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        cache_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5) if (user_temperature and not user_top_k) else kwargs.pop("top_k", -1),
        )
        if cache_data_list is None:
            cache_data_list = []
        cache_answers = []
        similarity_threshold = chat_cache.config.similarity_threshold
        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        for cache_data in cache_data_list:
            ret = chat_cache.data_manager.get_scalar_data(
                cache_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if ret is None:
                continue

            if "deps" in context and hasattr(ret.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None
                }
                eval_cache_data = {
                    "question": ret.question.deps[0].data,
                    "answer": ret.answers[0].answer,
                    "search_result": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": ret.question,
                    "answer": ret.answers[0].answer,
                    "search_result": cache_data,
                    "embedding": ret.embedding_data,
                }
            rank = chat_cache.similarity_evaluation.evaluation(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            if rank_threshold <= rank:
                cache_answers.append((rank, ret.answers[0].answer, cache_data))
                chat_cache.data_manager.hit_cache_callback(cache_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d[2]) for d in cache_answers)
        if len(cache_answers) != 0:
            if chat_cache.post_process_messages_func is temperature_softmax:
                return_message = chat_cache.post_process_messages_func(
                    messages=[t[1] for t in cache_answers],
                    scores=[t[0] for t in cache_answers],
                    temperature=temperature
                )
            else:
                return_message = chat_cache.post_process_messages_func(
                    [t[1] for t in cache_answers]
                )
            chat_cache.report.hint_cache()
            if session:
                chat_cache.data_manager.add_session(answers_dict[return_message], session.name, pre_embedding_data)
            return cache_data_convert(return_message)

    next_cache = chat_cache.next_cache
    if next_cache:
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        llm_data = llm_handler(*args, **kwargs)

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                chat_cache.data_manager.save(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )

            llm_data = update_cache_callback(llm_data, update_cache_func, *args, **kwargs)
        except Exception as e:  # pylint: disable=W0703
            logging.warning("failed to save the data to cache, error: %s", e)
    return llm_data
