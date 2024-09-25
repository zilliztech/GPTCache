import numpy as np

from gptcache.utils import import_voyageai
from gptcache.embedding.base import BaseEmbedding

import_voyageai()

import voyageai


class VoyageAI(BaseEmbedding):
    """Generate text embedding for given text using VoyageAI.

    :param model: The model name to use for generating embeddings. Defaults to 'voyage-3'.
    :type model: str
    :param api_key_path: The path to the VoyageAI API key file.
    :type api_key_path: str
    :param api_key: The VoyageAI API key. If it is None, the client will search for the API key in the following order:
                    1. api_key_path, path to the file containing the key;
                    2. environment variable VOYAGE_API_KEY_PATH, which can be set to the path to the file containing the key;
                    3. environment variable VOYAGE_API_KEY.
                    This behavior is defined by the VoyageAI Python SDK.
    :type api_key: str    
    :param input_type: The type of input data. Defaults to None. Default to None. Other options: query, document. 
                    More details can be found in the https://docs.voyageai.com/docs/embeddings
    :type input_type: str
    :param truncation: Whether to truncate the input data. Defaults to True.
    :type truncation: bool

    Example:
        .. code-block:: python

            from gptcache.embedding import VoyageAI

            test_sentence = 'Hello, world.'
            encoder = VoyageAI(model='voyage-3', api_key='your_voyageai_key')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "voyage-3", api_key_path: str = None, api_key: str = None, input_type: str = None, truncation: bool = True):
        voyageai.api_key_path = api_key_path
        voyageai.api_key = api_key

        self._vo = voyageai.Client()
        self._model = model
        self._input_type = input_type
        self._truncation = truncation

        if self._model in self.dim_dict():
            self.__dimension = self.dim_dict()[model]
        else:
            self.__dimension = None

    def to_embeddings(self, data, **_):
        """
        Generate embedding for the given text input.

        :param data: The input text.
        :type data: str or list[str]

        :return: The text embedding in the shape of (dim,).
        :rtype: numpy.ndarray
        """
        if not isinstance(data, list):
            data = [data]
        result = self._vo.embed(texts=data, model=self._model, input_type=self._input_type, truncation=self._truncation)
        embeddings = result.embeddings
        return np.array(embeddings).astype("float32").squeeze(0)

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
        return {"voyage-3": 1024,
                "voyage-3-lite": 512,
                "voyage-finance-2": 1024, 
                "voyage-multilingual-2": 1024,
                "voyage-law-2": 1024, 
                "voyage-code-2": 1536}
