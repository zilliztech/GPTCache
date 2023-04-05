from gptcache.utils import import_towhee
import_towhee()

import numpy as np
from towhee.dc2 import pipe, ops

class Towhee:
    # english model: paraphrase-albert-small-v2-onnx
    # chinese model: uer/albert-base-chinese-cluecorpussmall-onnx
    """Generate text embedding for given text using Towhee.

    Example:
        .. code-block:: python
        
            from gptcache.embedding import Towhee
            
            test_sentence = "Hello, world." 
            encoder = Towhee(model="paraphrase-albert-small-v2")
            embed = encoder.to_embeddings(test_sentence)
    """
    def __init__(self, model = "paraphrase-albert-small-v2-onnx"):
        if model == "paraphrase-albert-small-v2-onnx":
            self._pipe = (
                pipe.input('text')
                    .map('text', 'vec',
                        ops.towhee.paraphrase_albert_small_v2_onnx())
                    .map('vec', 'vec', ops.towhee.np_normalize())
                    .output('text', 'vec')
            )
        elif model == "uer/albert-base-chinese-cluecorpussmall-onnx":
            self._pipe = (
                pipe.input('text')
                    .map('text', 'vec',
                         ops.towhee.albert_base_chinese_onnx())
                    .map('vec', 'vec', ops.towhee.np_normalize())
                    .output('text', 'vec')
            )
        else:
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

    @property
    def dimension(self):
        return self.__dimension