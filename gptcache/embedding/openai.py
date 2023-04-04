import numpy as np
import openai


class OpenAI:
    @staticmethod
    def to_embeddings(data, **kwargs):
        sentence_embeddings = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=data
        )
        return np.array(sentence_embeddings["data"][0]["embedding"]).astype('float32')

    @staticmethod
    def dimension():
        return 1536
