import numpy as np
from gptcache.utils import import_sbert
from gptcache.embedding.base import BaseEmbedding

import_sbert()

from sentence_transformers import SentenceTransformer  # pylint: disable=C0413


class SBERT(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models of Sentence Transformers.

    :param model: model name, defaults to 'all-MiniLM-L6-v2'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import SBERT

            test_sentence = 'Hello, world.'
            encoder = SBERT('all-MiniLM-L6-v2')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.model.eval()
        self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        emb = self.model.encode(data).squeeze(0)

        if not self.__dimension:
            self.__dimension = len(emb)
        return np.array(emb).astype("float32")

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            self.__dimension = len(self.to_embeddings("foo"))
        return self.__dimension
