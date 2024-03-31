from tempfile import TemporaryDirectory
from typing import Any, Dict
from unittest.mock import patch

from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager import manager_factory
from gptcache.processor import ContextProcess
from gptcache.processor.pre import all_content
from gptcache.utils.response import get_message_from_openai_answer


class CITestContextProcess(ContextProcess):
    def __init__(self):
        self.content = ""

    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        self.content = all_content(data)

    def process_all_content(self) -> (Any, Any):
        save_content = self.content.upper()
        embedding_content = self.content
        return save_content, embedding_content


# def test_context_process():
#     with TemporaryDirectory(dir="./") as root:
#         map_manager = manager_factory(data_dir=root)
#         context_process = CITestContextProcess()
#         cache.init(
#             pre_embedding_func=context_process.pre_process, data_manager=map_manager
#         )

#         question = "test calculate 1+3"
#         expect_answer = "the result is 4"

#         cache.data_manager.save(question, expect_answer, cache.embedding_func(question))

#         from openai import OpenAI
#         response = openai.cache_openai_chat_complete(
#             OpenAI(
#                 api_key="API_KEY",
#             ),
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": question},
#             ],
#         )
#         answer_text = get_message_from_openai_answer(response)
#         assert answer_text == expect_answer, answer_text
#         cache.flush()

#         map_manager = manager_factory(data_dir=root)
#         content = f"You are a helpful assistant.\n{question}"
#         cache_answer = map_manager.search(content)[0]
#         assert cache_answer[0] == content.upper()
#         assert cache_answer[1].answer == expect_answer
#         assert cache_answer[2] == content
