import math

import numpy as np

from gptcache.similarity_evaluation.np import NumpyNormEvaluation


def test_norm():
    evaluation = NumpyNormEvaluation(enable_normal=True)

    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 2.0)

    score = evaluation.evaluation(
        {
            "embedding": np.array([-0.5, -0.5])
        },
        {
            "search_result": (0, np.array([1, 1]))
        }
    )
    assert math.isclose(score, 2.0), score

    score = evaluation.evaluation(
        {
            "embedding": np.array([1, 2, 3, 4])
        },
        {
            "search_result": (0, np.array([0.1, 0.2, 0.3, 0.4]))
        }
    )
    assert math.isclose(score, 0.0), score
