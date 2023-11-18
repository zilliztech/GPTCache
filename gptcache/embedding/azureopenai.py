from typing import Union

import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_openai

import_openai()

from openai.lib.azure import AzureOpenAI, AsyncAzureOpenAI


class AzureOpenAIEmbedding(BaseEmbedding):
    """Generate text embedding for given text using Azure's OpenAI service.

    :param model: OpenAI Client with any modifications you intend to use.
    :type model: str

    :param model: model id from the API, defaults to 'text-embedding-ada-002'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import AzureOpenAIEmbedding
            from openai import AzureOpenAI

            test_sentence = 'Hello, world.'
            client = AzureOpenAI()

            # You can create different deployments for different embedding models on Azure.
            encoder = OpenAIEmbedding(client, azure_deployment='my_embedding_azure_deployment')

            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self,
                 client: Union[AzureOpenAI, AsyncAzureOpenAI],
                 azure_deployment: str,
                 model: str = 'text-embedding-ada-002',
                 ):
        """

        :param client: Azure OpenAI Client class
        :type client: Union[AzureOpenAI, AsyncAzureOpenAI]
        :param azure_deployment: The deployment name for the embedding; used to generate the endpoint url.
        :type azure_deployment: str
        :param model: The name of the embedding model; only used for determining the dimensions.  Defaults to "text-embedding-ada-002"
        :type model: str
        """
        self.model = model
        self.__azure_embedding_model_deployment = azure_deployment
        self.__dimension = self.dim_dict().get(self.model, None)
        self.client = client

    def to_embeddings(self, data, **_):
        """

        :param data: String that you wish to convert to an embedding.
        :param _:
        :return: Array of Float32 numbers that represent the string
        """
        sentence_embeddings = self.client.embeddings.create(
            input=data,
            model=self.__azure_embedding_model_deployment,
        )
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
