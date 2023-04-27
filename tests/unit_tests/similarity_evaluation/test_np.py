import math

import numpy as np

from gptcache.adapter.api import _get_eval
from gptcache.similarity_evaluation import NumpyNormEvaluation


embedding_func = lambda x: np.array([1, 1])


def _test_evaluation(evaluation):

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
    assert math.isclose(score, 2.0), score

    score = evaluation.evaluation(
        {"question": "test1"},
        {"question": "test2"}
    )
    assert math.isclose(score, 2.0), score


def test_norm():
    evaluation = NumpyNormEvaluation(enable_normal=True, question_embedding_function=embedding_func)
    _test_evaluation(evaluation)


def test_get_eval():
    evaluation = _get_eval(strategy="numpy", kws={"enable_normal": True, "question_embedding_function": embedding_func})


if __name__ == "__main__":
    test_norm()
