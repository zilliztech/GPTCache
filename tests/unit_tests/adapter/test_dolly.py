import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_inputs

question = "test_dolly"
expect_answer = "hello world"
onnx = Onnx()


class MockDolly:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, inputs, **kwargs):
        return [{"generated_text": expect_answer}]


class TestDolly(unittest.TestCase):
    def test_normal(self):
        with patch('gptcache.utils.import_torch'), \
             patch('gptcache.utils.import_huggingface'), \
             patch('transformers.pipeline') as mock_pipeline:

            with TemporaryDirectory(dir="./") as root:
                m = manager_factory('sqlite,faiss,local', data_dir=root, vector_params={"dimension": onnx.dimension})
                llm_cache = Cache()
                llm_cache.init(
                    pre_embedding_func=get_inputs,
                    data_manager=m,
                    embedding_func=onnx.to_embeddings
                )
                
                from gptcache.adapter.dolly import Dolly

                mock_pipeline.return_value = MockDolly()
                dolly = Dolly.from_model('dolly_model')
                answer = dolly(question, cache_obj=llm_cache)
                self.assertEqual(answer[0]["generated_text"], expect_answer)
                self.assertFalse(answer[0].get("gptcache", False))
                answer = dolly(question, cache_obj=llm_cache)
                self.assertEqual(answer[0]["generated_text"], expect_answer)
                self.assertTrue(answer[0].get("gptcache", False))

            with TemporaryDirectory(dir="./") as root:
                m = manager_factory('sqlite,faiss,local', data_dir=root, vector_params={"dimension": onnx.dimension})
                llm_cache = Cache()
                llm_cache.init(
                    pre_embedding_func=get_inputs,
                    data_manager=m,
                    embedding_func=onnx.to_embeddings
                )
                
                from gptcache.adapter.dolly import Dolly
                from transformers import pipeline
                dolly = Dolly(pipeline('dolly'))
                answer = dolly(question, cache_obj=llm_cache)
                self.assertEqual(answer[0]["generated_text"], expect_answer)
                self.assertFalse(answer[0].get("gptcache", False))
                answer = dolly(question, cache_obj=llm_cache)
                self.assertEqual(answer[0]["generated_text"], expect_answer)
                self.assertTrue(answer[0].get("gptcache", False))                
