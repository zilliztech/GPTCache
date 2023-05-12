import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_langchain

import_langchain()

from langchain.embeddings.base import Embeddings  # pylint: disable=C0413


class LangChain(BaseEmbedding):
    """Generate text embedding for given text using LangChain

    :param embeddings: the LangChain Embeddings object.
    :type embeddings: Embeddings
    :param dimension: The vector dimension after embedding is calculated by calling embed once by default.
     If you confirm the dimension, you can assign a value to this parameter to reduce this request.
    :type dimension: int

    Example:
        .. code-block:: python

            from gptcache.embedding import LangChain
            from langchain.embeddings.openai import OpenAIEmbeddings

            test_sentence = 'Hello, world.'
            embeddings = OpenAIEmbeddings(model="your-embeddings-deployment-name")
            encoder = LangChain(embeddings=embeddings)
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, embeddings: Embeddings, dimension: int = 0):
        self._embeddings: Embeddings = embeddings
        self._dimension: int = (
            dimension if dimension != 0 else len(self._embeddings.embed_query("foo"))
        )

    def to_embeddings(self, data, **kwargs):
        vector_data = self._embeddings.embed_query(data)
        return np.array(vector_data).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension
