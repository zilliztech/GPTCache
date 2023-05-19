import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_huggingface

import_huggingface()

from transformers import AutoTokenizer, RwkvModel  # pylint: disable=C0413


class Rwkv(BaseEmbedding):
    """Generate sentence embedding for given text using RWKV models.

    :param model: model name, defaults to 'sgugger/rwkv-430M-pile'. Check
      https://huggingface.co/docs/transformers/model_doc/rwkv for more avaliable models.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import Rwkv

            test_sentence = 'Hello, world.'
            encoder = Rwkv(model='sgugger/rwkv-430M-pile')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "sgugger/rwkv-430M-pile"):
        self.model = RwkvModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:  # pylint: disable=W0703
            from transformers import AutoConfig  # pylint: disable=C0415

            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        inputs = self.tokenizer(data, return_tensors="pt")
        outputs = self.model(inputs["input_ids"])
        emb = outputs.last_hidden_state[0, 0, :].detach().numpy()
        return np.array(emb).astype("float32")

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
