from tempfile import TemporaryDirectory
from typing import Any, Dict
from unittest.mock import patch

from gptcache.manager import manager_factory
from gptcache.utils.response import get_message_from_openai_answer
from gptcache.adapter import openai
from gptcache import cache
from gptcache.processor.pre import all_content
from gptcache.processor import ContextProcess


class TestContextProcess(ContextProcess):
    def __init__(self):
        self.content = ""

    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        self.content = all_content(data)

    def process_all_content(self) -> (Any, Any):
        save_content = self.content.upper()
        embedding_content = self.content
        return save_content, embedding_content


def test_context_process():
    with TemporaryDirectory(dir="./") as root:
        map_manager = manager_factory(data_dir=root)
        context_process = TestContextProcess()
        cache.init(
            pre_embedding_func=context_process.pre_process, data_manager=map_manager
        )

        question = "test calculate 1+3"
        expect_answer = "the result is 4"
        with patch("openai.ChatCompletion.create") as mock_create:
            datas = {
                "choices": [
                    {
                        "message": {"content": expect_answer, "role": "assistant"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "created": 1677825464,
                "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "gpt-3.5-turbo-0301",
                "object": "chat.completion.chunk",
            }
            mock_create.return_value = datas

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
            )

            assert get_message_from_openai_answer(response) == expect_answer, response

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        answer_text = get_message_from_openai_answer(response)
        assert answer_text == expect_answer, answer_text
        cache.flush()

        map_manager = manager_factory(data_dir=root)
        content = f"You are a helpful assistant.\n{question}"
        cache_answer = map_manager.search(content)[0]
        assert cache_answer[0] == content.upper()
        assert cache_answer[1].answer == expect_answer
        assert cache_answer[2] == content
