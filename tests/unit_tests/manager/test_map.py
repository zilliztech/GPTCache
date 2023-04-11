import os
import unittest
import numpy as np

from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager.data_manager import MapDataManager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

data_map_path = "data_map_test_map.txt"


def mock_embeddings():
    return np.random.random((10,)).astype('float32')


data_manager = MapDataManager(data_map_path, 3)
cache.init(embedding_func=mock_embeddings,
           data_manager=data_manager,
           similarity_evaluation=SearchDistanceEvaluation(),
           )


class TestMapDataManager(unittest.TestCase):
    def test_map(self):
        a = "a"
        for i in range(4):
            data_manager.save(chr(ord(a) + i), str(i), chr(ord(a) + i))
        self.assertEqual(len(data_manager.search("a")), 0)
        question, answer = data_manager.search("b")[0]
        self.assertEqual(question, 'b')
        self.assertEqual(answer, '1')
        data_manager.close()

    @classmethod
    def tearDownClass(cls) -> None:
        answers = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': 'b'}
            ],
        )
        answer = answers["choices"][0]["message"]["content"]
        cls().assertEqual(answer, '1')

        os.remove(data_map_path)
