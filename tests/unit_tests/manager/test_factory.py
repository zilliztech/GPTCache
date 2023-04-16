import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from gptcache.manager.factory import get_data_manager
from gptcache.manager import VectorBase, CacheBase, ObjectBase
from gptcache.manager.scalar_data.base import Answer, AnswerType


class TestFactory(unittest.TestCase):
    def test_normal(self):
        m1 = get_data_manager("sqlite", "chromadb", "local")
        self.assertIsNotNone(m1)

        m2 = get_data_manager("sqlite", "chromadb")
        self.assertIsNotNone(m2)

    def test_manager(self):
        with TemporaryDirectory(dir="./") as root:
            index_path = Path(root) / "faiss.bin"
            v = VectorBase("faiss", dimension=5, index_path=str(index_path))
            
            sql_url = "sqlite:///" + str(Path(root) / "sqlite.db")
            s = CacheBase("sqlite", sql_url=sql_url)

            o = ObjectBase('local', path=str(root))
            m = get_data_manager(s, v, o)
            m.save("test_question",
                   Answer(b"my test data",
                          AnswerType.IMAGE_BASE64),
                   np.random.rand(5)
            )
            res = m.get_scalar_data(m.search(np.random.rand(5))[0])
            self.assertEqual(res.question, "test_question")
            self.assertEqual(res.answers[0].answer, b"my test data")
            self.assertEqual(res.answers[0].answer_type, AnswerType.IMAGE_BASE64)

            # test multi answer
            emb = np.random.rand(5)
            m.save("test_question2",
                   [Answer(b"question2_BASE64",
                           AnswerType.IMAGE_BASE64),
                    Answer("question2_STR",
                           AnswerType.STR)
                   ],
                   emb
            )

            res = m.get_scalar_data(m.search(emb)[0])
            self.assertEqual(res.question, "test_question2")
            self.assertEqual(res.answers[0].answer, b"question2_BASE64")
            self.assertEqual(res.answers[0].answer_type, AnswerType.IMAGE_BASE64)
            self.assertEqual(res.answers[1].answer, "question2_STR")
            self.assertEqual(res.answers[1].answer_type, AnswerType.STR)
