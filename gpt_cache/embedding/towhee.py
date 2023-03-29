import numpy as np
from towhee.dc2 import pipe, ops


class Towhee:
    # english model: paraphrase-albert-small-v2
    # chinese model: uer/albert-base-chinese-cluecorpussmall
    def __init__(self, model="paraphrase-albert-small-v2"):
        self._pipe = (
            pipe.input('text')
                .map('text', 'vec',
                     ops.sentence_embedding.transformers(model_name=model))
                .map('vec', 'vec', ops.towhee.np_normalize())
                .output('text', 'vec')
        )
        self.__dimension = len(self._pipe("foo").get_dict()['vec'])

    def to_embeddings(self, data, **kwargs):
        emb = self._pipe(data).get_dict()['vec']
        return np.array(emb).astype('float32')

    def dimension(self):
        return self.__dimension
