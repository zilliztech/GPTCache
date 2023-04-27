from gptcache.similarity_evaluation import KReciprocalEvaluation
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.manager.vector_data.base import VectorData
from gptcache.adapter.api import _get_eval
import numpy as np
import math

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm

faiss = Faiss('./none', 3, 10)


def _test_evaluation(evaluation):
    narr1 = normalize(np.array([1.0, 2.0, 3.0]))
    faiss.mul_add([VectorData(id=0, data=narr1)])
    narr2 = normalize(np.array([2.0, 3.0, 4.0]))
    faiss.mul_add([VectorData(id=1, data=narr2)])
    narr3 = normalize(np.array([3.0, 4.0, 5.0]))
    faiss.mul_add([VectorData(id=2, data=narr3)])
    evaluation = KReciprocalEvaluation(vectordb=faiss, top_k=2)
    query1 = normalize(np.array([1.1, 2.1, 3.1]))
    query2 = normalize(np.array([101.1, 102.1, 103.1]))

    score1 = evaluation.evaluation({'question': 'question1', 'embedding': query1}, {'question': 'question2', 'embedding': narr1})
    score2 = evaluation.evaluation({'question': 'question1', 'embedding': query2}, {'question': 'question2', 'embedding': narr1})

    assert score1 > 3.99
    assert math.isclose(score2, 0)

def test_kreciprocal():
    evaluation = KReciprocalEvaluation(vectordb=faiss, top_k=2)
    _test_evaluation(evaluation)

def test_get_eval():
    evaluation = _get_eval(strategy="kreciprocal", kws={"vectordb": faiss, "top_k": 2})
    _test_evaluation(evaluation)
