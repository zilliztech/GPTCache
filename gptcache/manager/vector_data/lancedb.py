from typing import List, Optional

import numpy as np
import pyarrow as pa
import lancedb
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

        # Initialize or open table
        if self._table_name not in self._db.table_names():
            self._table = None  # Table will be created with the first insertion
        else:
            self._table = self._db.open_table(self._table_name)

    def mul_add(self, datas: List[VectorData]):
        """Add multiple vectors to the LanceDB table"""
        vectors, vector_ids = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        # Infer the dimension of the vectors
        vector_dim = len(vectors[0]) if vectors else 0

        # Create table with the inferred schema if it doesn't exist
        if self._table is None:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=vector_dim))
            ])
            self._table = self._db.create_table(self._table_name, schema=schema)

        # Prepare and add data to the table
        self._table.add(({"id": vector_id, "vector": vector} for vector_id, vector in zip(vector_ids, vectors)))

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
        for vector_id in ids:
            self._table.delete(f"id = '{vector_id}'")

    def rebuild(self, ids: Optional[List[int]] = None):
        """Rebuild the index, if applicable"""
        return True

    def count(self):
        """Return the total number of vectors in the table"""
        return len(self._table)
