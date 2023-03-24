import numpy as np
from towhee.dc2 import pipe, ops, DataCollection


def to_embeddings(data, **kwargs):
    message = data.get("messages")[-1]["content"]
    p = (
        pipe.input('text')
            .map('text', 'vec', ops.sentence_embedding.transformers(model_name='paraphrase-albert-small-v2'))
            .map('vec', 'vec', ops.towhee.np_normalize())
            .output('text', 'vec')
    )

    emb = p(message).get_dict()['vec']
    # dimension 768
    return np.array(emb).astype('float32').reshape(1, -1)
