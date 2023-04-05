import numpy as np
import openai


class OpenAI:
    """Generate text embedding for given text using OpenAI.

    Example:
        .. code-block:: python

            from gptcache.embedding import openai
            
            test_sentence = "Hello, world." 
            encoder = openai(api_key="your_openai_key")
            embed = encoder.to_embeddings(test_sentence)
    """
    def __init__(self, api_key, model="text-embedding-ada-002", **kwargs):
        self.api_key = api_key
        self.model = model
    
    def to_embeddings(self, data):
        sentence_embeddings = openai.Embedding.create(
            api_key=self.api_key,
            model=self.model,
            input=data
        )
        return np.array(sentence_embeddings["data"][0]["embedding"]).astype('float32')

    @staticmethod
    def dimension():
        return 1536
