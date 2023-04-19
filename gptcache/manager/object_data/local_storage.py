from typing import Any, List
import os
import uuid
from pathlib import Path
from gptcache.manager.object_data.base import ObjectBase
from gptcache.utils.log import gptcache_log


class LocalObjectStorage(ObjectBase):
    """Local object storage
    """

    def __init__(self, local_root: str):
        self._local_root = Path(local_root)
        self._local_root.mkdir(exist_ok=True)

    def put(self, obj: Any) -> str:
        f_path = self._local_root / str(uuid.uuid4())
        with open(f_path, "wb") as f:
            f.write(obj)
        return str(f_path.absolute())

    def get(self, obj: str) -> Any:
        try:
            with open(obj, "rb") as f:
                return f.read()
        except Exception: # pylint: disable=broad-except
            return None

    def get_access_link(self, obj: str, _: int = 3600):
        return obj

    def delete(self, to_delete: List[str]):
        assert isinstance(to_delete, list)
        for obj in to_delete:
            try:
                os.remove(obj)
            except Exception:  # pylint: disable=broad-except
                gptcache_log.warning("Can not find obj: %s", obj)
                pass
