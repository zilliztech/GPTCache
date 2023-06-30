import math

from gptcache.adapter.api import _get_eval
from gptcache.similarity_evaluation import OnnxModelEvaluation


def _test_evaluation(evaluation):
    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 1.0)

    score = evaluation.evaluation({"question": "hello"}, {"question": "hello"})
    assert math.isclose(score, 1.0)

    query = "Can you pass a urine test for meth in 4 days?"
    candidate_1 = "Can meth be detected in a urine test if last used was Thursday night and the test was tuesday morning?"
    candidate_2 = "how old are you?"

    score = evaluation.evaluation({"question": query}, {"question": candidate_1})
    assert isinstance(score, float), type(score)
    assert score > 0.8

    score = evaluation.evaluation({"question": query}, {"question": candidate_2})
    assert score < 0.1


def test_onnx():
    evaluation = OnnxModelEvaluation()
    _test_evaluation(evaluation)


def test_get_eval():
    evaluation = _get_eval("onnx")
    _test_evaluation(evaluation)
