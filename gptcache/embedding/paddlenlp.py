import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_paddlenlp, import_paddle

import_paddle()
import_paddlenlp()


import paddle  # pylint: disable=C0413
from paddlenlp.transformers import AutoModel, AutoTokenizer  # pylint: disable=C0413

class PaddleNLP(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models from PaddleNLP transformers.

    :param model: model name, defaults to 'ernie-3.0-medium-zh'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import PaddleNLP

            test_sentence = 'Hello, world.'
            encoder = PaddleNLP(model='ernie-3.0-medium-zh')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "ernie-3.0-medium-zh"):
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "<pad>"
        self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pd"
        )
        outs = self.model(**inputs)[0]
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype("float32")

    def post_proc(self, token_embeddings, inputs):
        attention_mask = paddle.ones(inputs["token_type_ids"].shape)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).astype("float32")
        )
        sentence_embs = paddle.sum(
            token_embeddings * input_mask_expanded, 1
        ) / paddle.clip(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            self.__dimension = len(self.to_embeddings("foo"))
        return self.__dimension

