import uuid
from typing import Callable

from gptcache import cache
from gptcache.manager.data_manager import DataManager
from gptcache.utils.log import gptcache_log
from gptcache.processor.check_hit import check_hit_session


class Session:
    """"
    Session for gptcache.

    :param name: the name of the session, defaults to `uuid.uuid4().hex`.
    :type name:  str
    :param data_manager: the DataManager of the session, defaults to cache.data_manager with the initialized cache.
    :type data_manager: DataManager
    :param check_hit_func: a Callable to check the hit, defaults to `processor.check_hit.check_hit_session`，which will not return cached data
                           if you ask the same or similar question in the same session.
    :type check_hit_func:  Callable


    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.session import Session
            # init gptcache
            cache.init()
            cache.set_openai_key()
            session = Session()

            from gptcache.adapter import openai
            # run ChatCompletion model with gptcache on session
            response = openai.ChatCompletion.create(
                          model='gpt-3.5-turbo',
                          messages=[
                            {
                                'role': 'user',
                                'content': "what's github"
                            }],
                          session=session
                        )
            response_content = response['choices'][0]['message']['content']
    """
    def __init__(self, name: str = None, data_manager: DataManager = None, check_hit_func: Callable = None):
        self._name = uuid.uuid4().hex if not name else name
        self._data_manager = cache.data_manager if not data_manager else data_manager
        self.check_hit_func = check_hit_session if not check_hit_func else check_hit_func

    @property
    def name(self):
        return self._name

    def __enter__(self):
        gptcache_log.warning("The `with` method will delete the session data directly on exit.")
        return self

    def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
        self.drop()

    def drop(self):
        self._data_manager.delete_session(self.name)
        gptcache_log.info("Deleting data in the session: %s.", self.name)
