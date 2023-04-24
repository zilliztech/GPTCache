import math

import numpy as np

from gptcache.similarity_evaluation import NumpyNormEvaluation


def test_norm():
    embedding_func = lambda x: np.array([1, 1])
    evaluation = NumpyNormEvaluation(enable_normal=True, question_embedding_function=embedding_func)

    range_min, range_max = evaluation.range()
    # print(range_max)
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 2.0)

    score = evaluation.evaluation(
        {"embedding": np.array([-0.5, -0.5])}, {"embedding": np.array([1, 1])}
    )
    assert math.isclose(score, 0.0, abs_tol=0.001), score

    score = evaluation.evaluation(
        {"embedding": np.array([1, 2, 3, 4])},
        {"embedding": np.array([0.1, 0.2, 0.3, 0.4])},
    )

    assert math.isclose(score, 2.0, abs_tol=0.001), score

    score = evaluation.evaluation(
        {"question": "test"},
        {"question": "test"}
    )
    assert math.isclose(score, 0.0), score

    score = evaluation.evaluation(
        {"question": "test1"},
        {"question": "test2"}
    )
    assert math.isclose(score, 2.0), score


if __name__ == "__main__":
    test_norm()
