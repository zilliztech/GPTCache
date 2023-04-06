import math

from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


def test_search_distance_evaluation():
    evaluation = SearchDistanceEvaluation()

    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 4.0)

    score = evaluation.evaluation(
        {},
        {
            "search_result": (1, None)
        }
    )
    assert math.isclose(score, 3.0)

    score = evaluation.evaluation(
        {},
        {
            "search_result": (-1, None)
        }
    )
    assert math.isclose(score, 4.0)

    evaluation = SearchDistanceEvaluation(max_distance=10, positive=True)
    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 10.0)

    score = evaluation.evaluation(
        {},
        {
            "search_result": (5, None)
        }
    )
    assert math.isclose(score, 5.0)
    score = evaluation.evaluation(
        {},
        {
            "search_result": (20, None)
        }
    )
    assert math.isclose(score, 10.0)
