import time

import numpy as np

from gptcache import cache
from gptcache.processor.post import temperature_softmax
from gptcache.utils.error import NotInitError
from gptcache.utils.log import gptcache_log
from gptcache.utils.time import time_cal


def adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs):
    """Adapt to different llm

    :param llm_handler: LLM calling method, when the cache misses, this function will be called
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
    search_only_flag = kwargs.pop("search_only", False)
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
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
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
        for search_data in search_data_list:
            cache_data = time_cal(
                chat_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            # cache consistency check
            if chat_cache.config.data_check:
                is_healthy = cache_health_check(
                    chat_cache.data_manager.v,
                    {
                        "embedding": cache_data.embedding_data,
                        "search_result": search_data,
                    },
                )
                if not is_healthy:
                    continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                chat_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            hit_callback = kwargs.pop("hit_callback", None)
            if hit_callback and callable(hit_callback):
                factor = max_rank - min_rank
                hit_callback([(d[3].question, d[0] / factor if factor else d[0]) for d in cache_answers])
            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data and not chat_cache.config.disable_report:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3]
                report_search_data = cache_whole_data[2]
                chat_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return cache_data_convert(return_message)

    next_cache = chat_cache.next_cache
    if next_cache:
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        kwargs["search_only"] = search_only_flag
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        if search_only_flag:
            # cache miss
            return None
        llm_data = time_cal(
            llm_handler, func_name="llm_request", report_func=chat_cache.report.llm
        )(*args, **kwargs)

    if not llm_data:
        return None

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()

            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception as e:  # pylint: disable=W0703
            gptcache_log.warning("failed to save the data to cache, error: %s", e)
    return llm_data


async def aadapt(
    llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
):
    """Simple copy of the 'adapt' method to different llm for 'async llm function'

    :param llm_handler: Async LLM calling method, when the cache misses, this function will be called
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
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
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
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
        for search_data in search_data_list:
            cache_data = time_cal(
                chat_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                chat_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3]
                report_search_data = cache_whole_data[2]
                chat_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
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
        llm_data = await llm_handler(*args, **kwargs)

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()
            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception:  # pylint: disable=W0703
            gptcache_log.error("failed to save the data to cache", exc_info=True)
    return llm_data


_input_summarizer = None


def _summarize_input(text, text_length):
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from gptcache.processor.context.summarization_context import (
        SummarizationContextProcess,
    )

    global _input_summarizer
    if _input_summarizer is None:
        _input_summarizer = SummarizationContextProcess()
    summarization = _input_summarizer.summarize_to_sentence([text], text_length)
    return summarization


def cache_health_check(vectordb, cache_dict):
    """This function checks if the embedding
    from vector store matches one in cache store.
    If cache store and vector store are out of
    sync with each other, cache retrieval can
    be incorrect.
    If this happens, force the similary score
    to the lowerest possible value.
    """
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    emb_in_vec = vectordb.get_embeddings(data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        gptcache_log.critical("Cache Store and Vector Store are out of sync!!!")
        # 0: identical, inf: different
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        # self-healing by replacing entry
        # in the vec store with the one
        # from cache store by the same
        # entry_id.
        vectordb.update_embeddings(
            data_id,
            emb=cache_dict["embedding"],
        )
    return flag
