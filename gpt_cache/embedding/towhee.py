import numpy as np
from towhee.dc2 import pipe, ops


def pre_embedding(data, **kwargs):
    return data.get("messages")[-1]["content"]


class Towhee:
    # english model: paraphrase-albert-small-v2
    # chinese model: uer/albert-base-chinese-cluecorpussmall
    def __init__(self, model="paraphrase-albert-small-v2"):
        self.__pipe = (
            pipe.input('text')
                .map('text', 'vec',
                     ops.sentence_embedding.transformers(model_name=model))
                .map('vec', 'vec', ops.towhee.np_normalize())
                .output('text', 'vec')
        )
        self.__dimension = len(self.__pipe("foo").get_dict()['vec'])

    def to_embeddings(self, data, **kwargs):
        message = pre_embedding(data, **kwargs)
        emb = self.__pipe(message).get_dict()['vec']
        return np.array(emb).astype('float32').reshape(1, -1)

    def dimension(self):
        return self.__dimension
