from typing import Union

import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_openai

import_openai()

from openai import OpenAI, AsyncOpenAI


class OpenAIEmbedding(BaseEmbedding):
    """Generate text embedding for given text using OpenAI.

    :param model: OpenAI Client with any modifications you intend to use.
    :type model: str

    :param model: model id from the API, defaults to 'text-embedding-ada-002'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import OpenAIEmbedding
            from openai import OpenAI

            test_sentence = 'Hello, world.'
            client = OpenAI(api_key='your_openai_key')
            encoder = OpenAIEmbedding(client, model="MyEmbeddingModelId")
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self,
                 client: Union[OpenAI, AsyncOpenAI],
                 model: str = "text-embedding-ada-002",
                 ):
        self.client = client
        self.model = model
        if model in self.dim_dict():
            self.__dimension = self.dim_dict()[model]
        else:
            self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        sentence_embeddings = await self.client.embeddings.create(model=self.model, input=data)
        return np.array(sentence_embeddings.data[0].embedding).astype("float32")

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            foo_emb = self.to_embeddings("foo")
            self.__dimension = len(foo_emb)
        return self.__dimension

    @staticmethod
    def dim_dict():
        return {"text-embedding-ada-002": 1536}
