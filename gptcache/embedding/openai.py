import numpy as np
import openai
import os


class OpenAI:
    """Generate text embedding for given text using OpenAI.

    Example:
        .. code-block:: python

            from gptcache.embedding import OpenAI
            
            test_sentence = "Hello, world." 
            encoder = OpenAI(api_key="your_openai_key")
            embed = encoder.to_embeddings(test_sentence)
    """
    def __init__(self, model: str="text-embedding-ada-002", api_key: str=None, **kwargs):
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
    
    def to_embeddings(self, data):
        sentence_embeddings = openai.Embedding.create(
            model=self.model,
            input=data
        )
        return np.array(sentence_embeddings["data"][0]["embedding"]).astype('float32')

    @property
    def dimension(self):
        if not self.__dimension:
            foo_emb = self.to_embeddings("foo")
            self.__dimension = len(foo_emb)
        return self.__dimension
    
    @staticmethod
    def dim_dict():
        return {
            "text-embedding-ada-002": 1536
        }
