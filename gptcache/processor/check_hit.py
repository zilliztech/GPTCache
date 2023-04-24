# pylint: disable=unused-argument
def check_hit_session(cur_session_id: str, cache_session_ids: list, cache_questions: list, cache_answer: str):
    return cur_session_id not in cache_session_ids
