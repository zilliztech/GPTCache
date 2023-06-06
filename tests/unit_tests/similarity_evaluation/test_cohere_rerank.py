import os
from unittest.mock import patch

from gptcache.adapter.api import _get_eval
from gptcache.utils import import_cohere

import_cohere()

from cohere.responses import Reranking


def test_cohere_rerank():
    os.environ["CO_API_KEY"] = "API"

    evaluation = _get_eval("cohere")

    min_value, max_value = evaluation.range()
    assert min_value < 0.001
    assert max_value > 0.999

    with patch("cohere.Client.rerank") as mock_create:
        mock_create.return_value = Reranking(
            response={
                "meta": {"api_version": {"version": "2022-12-06"}},
                "results": [],
            }
        )
        evaluation = _get_eval("cohere")
        score = evaluation.evaluation(
            {"question": "What is the color of sky?"},
            {"answer": "the color of sky is blue"},
        )
        assert score < 0.01

    with patch("cohere.Client.rerank") as mock_create:
        mock_create.return_value = Reranking(
            response={
                "meta": {"api_version": {"version": "2022-12-06"}},
                "results": [
                    {
                        "relevance_score": 0.9871293,
                        "index": 0,
                    }
                ],
            }
        )
        evaluation = _get_eval("cohere")
        score = evaluation.evaluation(
            {"question": "What is the color of sky?"},
            {"answer": "the color of sky is blue"},
        )
        assert score > 0.9
