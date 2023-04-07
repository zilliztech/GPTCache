import numpy as np
import openai
import os

from gptcache.embedding.base import BaseEmbedding


class OpenAI(BaseEmbedding):
    """Generate text embedding for given text using OpenAI.

    :param model: model name, defaults to 'text-embedding-ada-002'.
    :type model: str
    :param api_key: OpenAI API Key. When the parameter is not specified, it will load the key by default if it is available.
    :type api_key: str

    Example:
        .. code-block:: python

            from gptcache.embedding import OpenAI

            test_sentence = 'Hello, world.'
            encoder = OpenAI(api_key='your_openai_key')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        if not api_key:
            if openai.api_key:
                api_key = openai.api_key
            else:
                api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
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
        sentence_embeddings = openai.Embedding.create(model=self.model, input=data)
        return np.array(sentence_embeddings["data"][0]["embedding"]).astype("float32")

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
