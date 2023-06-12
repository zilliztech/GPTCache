from gptcache.similarity_evaluation import SequenceMatchEvaluation
from gptcache.similarity_evaluation.sequence_match import reweight
from gptcache.adapter.api import _get_eval
import numpy as np
import math

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm

def _test_evaluation(evaluation):
    evaluation = SequenceMatchEvaluation([0.1, 0.2, 0.7], 'onnx')
    score1 = evaluation.evaluation({'question': 'USER:foo1\nUSER:foo2\nUSER:foo3\n'}, {'question': 'USER:foo1\nUSER:foo2\nUSER:foo3\n'})
    score2 = evaluation.evaluation({'question': 'USER:foo1\nUSER:foo2\nUSER:foo3\n'}, {'question': 'USER:foo1\nUSER:foo2\n'})
    evaluation = SequenceMatchEvaluation([0.2, 0.8], 'onnx')
    score2 = evaluation.evaluation({'question': 'USER:foo1\nUser:foo2\nUser:foo3\n'}, {'question': 'USER:foo1\nUser:foo2\n'})
    assert True

def test_sequence_match():
    evaluation = SequenceMatchEvaluation([0.1, 0.2, 0.7], 'onnx')
    evaluation.range()
    _test_evaluation(evaluation)

def test_get_eval():
    evaluation = _get_eval(strategy="sequence_match", kws={"embedding_extractor":'onnx', "weights": [0.1, 0.2, 0.7]})
    _test_evaluation(evaluation)

def test_reweigth():
    ws = reweight([0.7,0.2,0.1], 4)
    assert len(ws) == 3
    ws = reweight([0.7,0.2,0.1], 3)
    assert len(ws) == 3
    ws = reweight([0.7,0.2,0.1], 2)
    assert len(ws) == 2
    ws = reweight([0.7,0.2,0.1], 1)
    assert len(ws) == 1 

if __name__ == '__main__':
    test_sequence_match()
