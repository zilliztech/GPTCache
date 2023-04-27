import unittest
from unittest.mock import patch
from openai.error import AuthenticationError

from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager import manager_factory
from gptcache.session import Session
from gptcache.processor.pre import get_prompt
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.utils.response import get_text_from_openai_answer


def check_hit(cur_session_id, cache_session_ids, cache_questions, cache_answer):
    if cache_questions and "what" in cache_questions[0]:
        return True
    return False


class TestSession(unittest.TestCase):
    """Test Session"""
    question = "what is your name?"
    expect_answer = "gptcache"
    session_id = "test_map"

    def test_with(self):
        data_manager = manager_factory("map", data_dir="./test_session")
        cache.init(data_manager=data_manager, pre_embedding_func=get_prompt)

        session0 = Session(self.session_id, check_hit_func=check_hit)
        self.assertEqual(session0.name, self.session_id)
        with patch("openai.Completion.create") as mock_create:
            mock_create.return_value = {
                "choices": [{"text": self.expect_answer, "finish_reason": None, "index": 0}],
                "created": 1677825464,
                "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "text-davinci-003",
                "object": "text_completion",
            }

            with Session() as session:
                response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session)
                answer_text = get_text_from_openai_answer(response)
                self.assertEqual(answer_text, self.expect_answer)
        self.assertEqual(len(data_manager.list_sessions()), 0)

    def test_map(self):
        data_manager = manager_factory("map", data_dir="./test_session")
        cache.init(data_manager=data_manager, pre_embedding_func=get_prompt)

        session0 = Session(self.session_id, check_hit_func=check_hit)
        with patch("openai.Completion.create") as mock_create:
            mock_create.return_value = {
                "choices": [{"text": self.expect_answer, "finish_reason": None, "index": 0}],
                "created": 1677825464,
                "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "text-davinci-003",
                "object": "text_completion",
            }

            response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session0)
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

        response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session0)
        answer_text = get_text_from_openai_answer(response)
        self.assertEqual(answer_text, self.expect_answer)

        session1 = Session()
        response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session1)
        answer_text = get_text_from_openai_answer(response)
        self.assertEqual(answer_text, self.expect_answer)

        with self.assertRaises(AuthenticationError):
            openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session1)

        self.assertEqual(len(data_manager.list_sessions()), 2)
        session0.drop()
        session1.drop()
        self.assertEqual(len(data_manager.list_sessions()), 0)

    def test_ssd(self):
        onnx = Onnx()
        data_manager = manager_factory("sqlite,faiss", './test_session', vector_params={"dimension": onnx.dimension})
        cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )

        session0 = Session(self.session_id, check_hit_func=check_hit)
        with patch("openai.Completion.create") as mock_create:
            mock_create.return_value = {
                "choices": [{"text": self.expect_answer, "finish_reason": None, "index": 0}],
                "created": 1677825464,
                "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "text-davinci-003",
                "object": "text_completion",
            }

            response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session0)
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

        response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session0)
        answer_text = get_text_from_openai_answer(response)
        self.assertEqual(answer_text, self.expect_answer)

        session1 = Session()
        response = openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session1)
        answer_text = get_text_from_openai_answer(response)
        self.assertEqual(answer_text, self.expect_answer)

        with self.assertRaises(AuthenticationError):
            openai.Completion.create(model="text-davinci-003", prompt=self.question, session=session1)

        self.assertEqual(len(data_manager.list_sessions()), 2)
        session0.drop()
        session1.drop()
        self.assertEqual(len(data_manager.list_sessions()), 0)
