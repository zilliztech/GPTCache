import lancedb
import numpy as np
import pyarrow as pa
from typing import List, Optional
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_lancedb, import_torch


import_torch()
import_lancedb()


class LanceDB(VectorBase):
    """Vector store: LanceDB

    :param persist_directory: The directory to persist, defaults to '/tmp/lancedb'.
    :type persist_directory: str
    :param table_name: The name of the table in LanceDB, defaults to 'gptcache'.
    :type table_name: str
    :param top_k: The number of the vectors results to return, defaults to 1.
    :type top_k: int
    """

    def __init__(
        self,
        persist_directory: Optional[str] = "/tmp/lancedb",
        table_name: str = "gptcache",
        top_k: int = 1,
    ):
        self._persist_directory = persist_directory
        self._table_name = table_name
        self._top_k = top_k

        # Initialize LanceDB database
        self._db = lancedb.connect(self._persist_directory)

        # Define the schema if creating a new table
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=10))  # Assuming dimension 10 for vectors
        ])

        # Initialize or open table
        if self._table_name not in self._db.table_names():
            self._table = self._db.create_table(self._table_name, schema=schema)
        else:
            self._table = self._db.open_table(self._table_name)

    def mul_add(self, datas: List[VectorData]):
        """Add multiple vectors to the LanceDB table"""
        vectors, ids = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        data = [{"id": id, "vector": vector} for id, vector in zip(ids, vectors)]
        self._table.add(data)

    def search(self, data: np.ndarray, top_k: int = -1):
        """Search for the most similar vectors in the LanceDB table"""
        if len(self._table) == 0:
            return []

        if top_k == -1:
            top_k = self._top_k

        results = self._table.search(data.tolist()).limit(top_k).to_list()
        return [(result["_distance"], int(result["id"])) for result in results]

    def delete(self, ids: List[int]):
        """Delete vectors from the LanceDB table based on IDs"""
        for id in ids:
            self._table.delete(f"id = '{id}'")

    def rebuild(self, ids: Optional[List[int]] = None):  
        """Rebuild the index, if applicable"""
        return True

    def flush(self):
        """Flush changes to disk (if necessary)"""
        pass

    def close(self):
        """Close the connection to LanceDB"""
        pass

    def count(self):
        """Return the total number of vectors in the table"""
        return len(self._table)