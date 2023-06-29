import os
import shutil
from tempfile import TemporaryDirectory

import pytest

from base.client_base import Base
from common import common_func as cf
from gptcache import cache, Config
from gptcache.adapter import openai
from gptcache.embedding import SBERT
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


def get_text_response(response):
    if response is None:
        return ""
    collated_response = [
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in response
    ]
    return "".join(collated_response)


class TestSqliteMilvus(Base):

    """
    ******************************************************************
    #  The followings are general cases
    ******************************************************************
    """
    @pytest.mark.tags("L1")
    def test_cache_health_check(self):
        """
        target: test hit the cache function
        method: keep default similarity_threshold
        expected: cache health detection & correction
        """
        with TemporaryDirectory(dir="./") as root:

            onnx = SBERT()

            vector_bases = [
                VectorBase(
                    "milvus",
                    dimension=onnx.dimension,
                    local_mode=True,
                    port="10086",
                    local_data=str(root),
                ),
                VectorBase("chromadb"),
            ]

            for vector_base in vector_bases:
                if os.path.isfile("./sqlite.db"):
                    os.remove("./sqlite.db")
                if os.path.isdir('./milvus_data'):
                    shutil.rmtree('./milvus_data')

                data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
                cache.init(
                    embedding_func=onnx.to_embeddings,
                    data_manager=data_manager,
                    similarity_evaluation=SearchDistanceEvaluation(),
                    config=Config(
                        log_time_func=cf.log_time_func,
                        enable_token_counter=False,
                    ),
                )

                question = [
                    "what is apple?",
                    "what is intel?",
                    "what is openai?",

                ]
                answer = [
                    "apple",
                    "intel",
                    "openai"
                ]
                for q, a in zip(question, answer):
                    cache.data_manager.save(q, a, cache.embedding_func(q))

                # let's simulate cache out-of-sync
                # situation.
                touble_query = "what is google?"
                cache.data_manager.v.update_embeddings(1, cache.embedding_func(touble_query))

                # without cache health check
                # respons is incorrect
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": touble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                # Incorrect response "apple" returned to user
                resp_txt = get_text_response(response)
                # log.info(f"Inccorect response = {resp_txt} is returned")
                assert answer[0] == resp_txt

                # cache health enabled
                # stop returning incorrect answer
                # and self-heal the trouble cache
                # entry.
                cache.config.data_check = True
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": touble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert response is None

                # disable cache check, and verify
                # cache is now consistent
                cache.config.data_check = False
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": touble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert response is None

                # verify self-heal took place
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question[0]},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert get_text_response(response) == answer[0]
                if os.path.isfile("./sqlite.db"):
                    os.remove("./sqlite.db")
