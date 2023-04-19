import math

from gptcache.similarity_evaluation import ExactMatchEvaluation


def test_exact_match_evaluation():
    evaluation = ExactMatchEvaluation()

    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 1.0)

    score = evaluation.evaluation({"question": "hello"}, {"question": "hello"})
    assert math.isclose(score, 1.0)

    score = evaluation.evaluation({"question": "tello"}, {"question": "hello"})
    assert math.isclose(score, 0.0)
