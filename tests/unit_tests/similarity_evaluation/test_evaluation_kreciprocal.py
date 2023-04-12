from gptcache.similarity_evaluation import KReciprocalEvaluation
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.manager.vector_data.base import VectorData
import numpy as np
import math

def test_kreciprocal():
    faiss = Faiss('./none', 3, 10)
    narr1 = np.array([1.0, 2.0, 3.0])
    faiss.mul_add([VectorData(id=0, data=narr1)])
    narr2 = np.array([2.0, 3.0, 4.0])
    faiss.mul_add([VectorData(id=1, data=narr2)])
    narr3 = np.array([3.0, 4.0, 5.0])
    faiss.mul_add([VectorData(id=2, data=narr3)])
    evaluation = KReciprocalEvaluation(vectordb=faiss)
    query1 = np.array([1.1, 2.1, 3.1])
    query2 = np.array([101.1, 102.1, 103.1])

    score1 = evaluation.evaluation({'question': 'question1', 'embedding_data': query1}, {'question': 'question2', 'embedding_data': narr1})
    score2 = evaluation.evaluation({'question': 'question1', 'embedding_data': query2}, {'question': 'question2', 'embedding_data': narr1})

    assert math.isclose(score1, 3.97)
    assert math.isclose(score2, 4.00)

