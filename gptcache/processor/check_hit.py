# pylint: disable=unused-argument
def check_hit_session(cur_session_id: str, cache_session_ids: list, cache_questions: list, cache_answer: str):
    """
    Check if the sesion result meets the hit requirement.

    :param cur_session_id: the name of the current session.
    :type cur_session_id: str
    :param cache_session_ids: a list of session names for caching the same content if you are using map as a data management method.
                              Otherwise a list of session names for similar content and same answer.
    :type cache_session_ids: list
    :param cache_question: a list with one question which same as the you asked if you use a map as a data management method.
                           Otherwise it is a list that is similar to the question you asked with the same answer,
                           and it is correspondence with cache_session_ids.
    :type cache_question: list
    :param cache_answer: the content of the cached answer.
    :param cache_answer: str

    :return: True or False
    """
    return cur_session_id not in cache_session_ids
