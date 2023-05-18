import uuid
from typing import Callable, Optional

from gptcache import cache
from gptcache.manager.data_manager import DataManager
from gptcache.processor.check_hit import check_hit_session
from gptcache.utils.log import gptcache_log


class Session:
    """
    Session for gptcache. Session can isolate the context of each connection, and can also filter the results after recall,
    and if not satisfied will re-request rather than return the cache results directly.

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

    def __init__(
        self,
        name: Optional[str] = None,
        data_manager: Optional[DataManager] = None,
        check_hit_func: Optional[Callable] = None,
    ):
        self._name = uuid.uuid4().hex if not name else name
        self._data_manager = cache.data_manager if not data_manager else data_manager
        self.check_hit_func = (
            check_hit_session if not check_hit_func else check_hit_func
        )

    @property
    def name(self):
        return self._name

    def __enter__(self):
        gptcache_log.warning(
            "The `with` method will delete the session data directly on exit."
        )
        return self

    def __exit__(self, *_):
        self.drop()

    def drop(self):
        """Drop the session and delete all data in the session"""
        self._data_manager.delete_session(self.name)
        gptcache_log.info("Deleting data in the session: %s.", self.name)
