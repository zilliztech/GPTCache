"""Nomic embedding integration"""

import numpy as np

from gptcache.utils import import_nomic
from gptcache.embedding.base import BaseEmbedding

import_nomic()

# import nomic # pylint: disable=C0413
from nomic import cli # pylint: disable=C0413
from nomic import embed # pylint: disable=C0413

class Nomic(BaseEmbedding):
    """Generate text embedding for given text using Cohere.
    """
    def __init__(self,
                 model: str = "nomic-embed-text-v1.5",
                 api_key: str = None,
                 task_type: str = "search_document",
                 dimensionality: int = None) -> None:
        """Generate text embedding for given text using Nomic embed.

        :param model: model name, defaults to 'nomic-embed-text-v1.5'.
        :type model: str
        :param api_key: Nomic API Key.
        :type api_key: str
        :param task_type: Task type in Nomic, defaults to 'search_document'.
        :type task_type: str
        :param dimensionality: Desired dimension of embeddings.
        :type dimensionality: int

        Example:
        .. code-block:: python

            import os
            from gptcache.embedding import Nomic

            test_sentence = 'Hey this is Nomic embedding integration to gaptcache.'
            encoder = Nomic(model='nomic-embed-text-v1.5',
                            api_key=os.getenv("NOMIC_API_KEY"),
                            dimensionality=64)
            embed = encoder.to_embeddings(test_sentence)
        """
        # Login to nomic
        cli.login(token=api_key)

        self._model = model
        self._task_type = task_type
        self._dimensionality = dimensionality

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (self._dimensionality,).
        """
        if not isinstance(data, list):
            data = [data]

        # Response will be a dictionary with key 'embeddings'
        # and value will be a list of lists
        response = embed.text(
            texts=data,
            model=self._model,
            task_type=self._task_type,
            dimensionality=self._dimensionality)
        embeddings = response["embeddings"]
        return np.array(embeddings).astype("float32").squeeze(0)

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self._dimensionality:
            foo_emb = self.to_embeddings("foo")
            self._dimensionality = len(foo_emb)
        return self._dimensionality
