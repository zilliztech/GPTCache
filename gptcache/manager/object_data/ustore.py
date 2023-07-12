from typing import Any, List

from gptcache.manager.object_data.base import ObjectBase
from gptcache.utils.log import gptcache_log
from gptcache.utils import import_ustore

import_ustore()
from ustore import ucset
from ustore import rocksdb

class UStore(ObjectBase):
    """object storage: UStore

    :param type: the engine type of 'UStore', the default value is 'UCSet'.
    :type type: str
    :param config: the engine config of 'UStore', the default value is None for 'UCSet'.
    :type config: str
    """

    def __init__(
        self,
        type: str = "UCSet",
        config: str = None
    ):
        if type.lower() == "ucset":
            self._db = ucset.DataBase()
        elif type.lower() == "rocksdb":
            self._db = rocksdb.DataBase(config)
        else:
            self._db = None
            gptcache_log.error("Unknown 'UStore' engine type: %s", type)
            assert False, "Unknown engine type of 'UStore'"

    def put(self, obj: Any) -> str:
        unique_key = abs(hash(obj))
        self._db.main[unique_key] = obj
        return str(unique_key)

    def get(self, obj: str) -> Any:
        return self._db.main[int(obj)]

    def get_access_link(self, obj: str) -> str:
        return obj

    def delete(self, to_delete: List[str]):
        for key in to_delete:
            del self._db.main[int(key)]
