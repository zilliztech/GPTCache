class EvictionManager:
    """
    EvictionManager to manager the eviction policy.

    :param scalar_storage: CacheStorage to manager the scalar data.
    :type scalar_storage: :class:`CacheStorage`
    :param vector_base: VectorBase to manager the vector data.
    :type vector_base:  :class:`VectorBase`
    """

    MAX_MARK_COUNT = 5000
    MAX_MARK_RATE = 0.1
    BATCH_SIZE = 100000
    REBUILD_CONDITION = 5

    def __init__(self, scalar_storage, vector_base):
        self._scalar_storage = scalar_storage
        self._vector_base = vector_base
        self.delete_count = 0

    def check_evict(self):
        mark_count = self._scalar_storage.count(state=-1)
        all_count = self._scalar_storage.count(is_all=True)
        if (
            mark_count > self.MAX_MARK_COUNT
            or mark_count / all_count > self.MAX_MARK_RATE
        ):
            return True
        return False

    def delete(self):
        mark_ids = self._scalar_storage.get_ids(deleted=True)
        self._scalar_storage.clear_deleted_data()
        self._vector_base.delete(mark_ids)
        self.delete_count += 1
        if self.delete_count >= self.REBUILD_CONDITION:
            self.rebuild()

    def rebuild(self):
        self._scalar_storage.clear_deleted_data()
        ids = self._scalar_storage.get_ids(deleted=False)
        self._vector_base.rebuild(ids)
        self.delete_count = 0

    def soft_evict(self, marked_keys):
        self._scalar_storage.mark_deleted(marked_keys)
