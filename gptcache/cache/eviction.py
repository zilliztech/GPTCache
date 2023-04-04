import numpy as np


class EvictionManager:
    MAX_MARK_COUNT = 5000
    MAX_MARK_RATE = 0.1
    BATCH_SIZE = 100000

    def __init__(self, scalar_storage, vector_base):
        self._scalar_storage = scalar_storage
        self._vector_base = vector_base

    def check_evict(self):
        mark_count = self._scalar_storage.count(state=-1)
        all_count = self._scalar_storage.count(is_all=True)
        if mark_count > self.MAX_MARK_COUNT or mark_count / all_count > self.MAX_MARK_RATE:
            return True
        return False

    def delete(self):
        mark_ids = self._scalar_storage.get_ids_by_state(state=-1)
        mark_ids = [i[0] for i in mark_ids]
        self._scalar_storage.remove_by_state()
        self._vector_base.delete(mark_ids)

    def rebuild(self):
        self._scalar_storage.remove_by_state()
        count = self._scalar_storage.count()
        offset = 0
        while offset < count:
            data = self._scalar_storage.get_embedding_data(offset, self.BATCH_SIZE)
            np_data = [np.frombuffer(d[0], np.float32) for d in data]
            self._vector_base.rebuild(np_data)
            offset += self.BATCH_SIZE

    def soft_evict(self, count):
        marked_keys = self._scalar_storage.get_old_access(count)
        marked_keys = [i[0] for i in marked_keys]
        self._scalar_storage.update_state(marked_keys)
