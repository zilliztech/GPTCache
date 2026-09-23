"""Segmented LRU (SLRU) cache with probation and protected segments.

New items enter probation. On a hit in probation, items are promoted to
protected. When protected overflows, its LRU victim is demoted back to
probation. Eviction victims are always taken from probation's LRU end.
"""

from collections import OrderedDict
from typing import Any, Optional, Tuple


class SegmentedLRU:
    """Two-segment LRU: probation (trial) + protected (proven).

    :param probation_capacity: max entries in probation segment
    :param protected_capacity: max entries in protected segment
    """

    def __init__(self, probation_capacity: int, protected_capacity: int):
        self._probation = OrderedDict()  # LRU: first item = least recent
        self._protected = OrderedDict()
        self._probation_cap = max(probation_capacity, 1)
        self._protected_cap = max(protected_capacity, 1)

    def get(self, key: Any) -> Optional[Any]:
        """Access a key, triggering promotion if in probation.

        Returns the value or None if not found.
        """
        # Hit in protected: refresh LRU position
        if key in self._protected:
            self._protected.move_to_end(key)
            return self._protected[key]

        # Hit in probation: promote to protected
        if key in self._probation:
            value = self._probation.pop(key)
            self._protected[key] = value
            self._protected.move_to_end(key)
            # If protected overflows, demote its LRU to probation
            self._handle_protected_overflow()
            return value

        return None

    def put(self, key: Any, value: Any = True):
        """Insert into probation segment (for newly admitted items)."""
        if key in self._protected:
            self._protected.move_to_end(key)
            self._protected[key] = value
            return
        if key in self._probation:
            self._probation.move_to_end(key)
            self._probation[key] = value
            return
        self._probation[key] = value
        self._probation.move_to_end(key)

    def peek_victim(self) -> Optional[Any]:
        """Peek at the probation LRU victim without removing it."""
        if not self._probation:
            return None
        return next(iter(self._probation))

    def evict(self) -> Optional[Tuple[Any, Any]]:
        """Remove and return the probation LRU victim as (key, value)."""
        if not self._probation:
            return None
        return self._probation.popitem(last=False)

    def remove(self, key: Any) -> bool:
        """Remove a key from either segment. Returns True if found."""
        if key in self._protected:
            del self._protected[key]
            return True
        if key in self._probation:
            del self._probation[key]
            return True
        return False

    def __contains__(self, key: Any) -> bool:
        return key in self._protected or key in self._probation

    def __len__(self) -> int:
        return len(self._protected) + len(self._probation)

    @property
    def probation_size(self) -> int:
        return len(self._probation)

    @property
    def protected_size(self) -> int:
        return len(self._protected)

    def _handle_protected_overflow(self):
        while len(self._protected) > self._protected_cap:
            demoted_key, demoted_val = self._protected.popitem(last=False)
            self._probation[demoted_key] = demoted_val
            self._probation.move_to_end(demoted_key)
