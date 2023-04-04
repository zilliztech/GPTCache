from .similarity_evaluation import SimilarityEvaluation
from ..util import import_towhee
import_towhee()

from towhee.dc2 import ops, pipe


class Towhee(SimilarityEvaluation):
    def __init__(self):
        self._pipe = (
            pipe.input('text', 'candidate')
                .map(('text', 'candidate'), 'similarity', ops.towhee.albert_duplicate())
                .output('similarity')
        )

    # WARNING: the model cannot evaluate text with more than 512 tokens
    def evaluation(self, src_dict, cache_dict, **kwargs):
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            return self._pipe(src_question, [cache_question]).get_dict()['similarity'][0]
        except Exception:
            return 0

    def range(self):
        return 0.0, 1.0
