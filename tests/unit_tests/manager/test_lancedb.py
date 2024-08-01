import unittest
import numpy as np
from gptcache.manager import VectorBase
from gptcache.manager.vector_data.base import VectorData

class TestLanceDB(unittest.TestCase):
    def test_normal(self):

        db = VectorBase("lancedb", persist_directory="/tmp/test_lancedb", top_k=3)
        
        # Add 100 vectors to the LanceDB
        db.mul_add([VectorData(id=i, data=np.random.sample(10)) for i in range(100)])
        
        # Perform a search with a random query vector
        search_res = db.search(np.random.sample(10))
        
        # Check that the search returns 3 results
        self.assertEqual(len(search_res), 3)
        
        # Delete vectors with specific IDs
        db.delete([1, 3, 5, 7])
        
        # Check that the count of vectors in the table is now 96
        self.assertEqual(db.count(), 96)