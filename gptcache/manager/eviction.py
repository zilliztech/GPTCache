class EvictionManager:
    """
    EvictionManager to manager the eviction policy.

    :param scalar_storage: CacheStorage to manager the scalar data.
    :type scalar_storage: :class:`CacheStorage`
    :param vector_base: VectorBase to manager the vector data.
    :type vector_base:  :class:`VectorBase`
    :param policy: The eviction policy, it is support "LRU" and "FIFO" now, and defaults to "LRU".
    :type policy:  str
    """

    MAX_MARK_COUNT = 5000
    MAX_MARK_RATE = 0.1
    BATCH_SIZE = 100000

    def __init__(self, scalar_storage, vector_base, policy="LRU"):
        self._scalar_storage = scalar_storage
        self._vector_base = vector_base
        self._policy = policy

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

    def rebuild(self):
        self._scalar_storage.clear_deleted_data()
        ids = self._scalar_storage.get_ids(deteted=False)
        self._vector_base.rebuild(ids)

    def soft_evict(self, count):
        if self._policy == "FIFO":
            marked_keys = self._scalar_storage.get_old_create(count)
        else:
            marked_keys = self._scalar_storage.get_old_access(count)
        self._scalar_storage.mark_deleted(marked_keys)
