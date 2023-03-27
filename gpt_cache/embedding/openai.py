import openai


class OpenAI:
    @staticmethod
    def to_embeddings(data, **kwargs):
        sentence_embeddings = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=data
        )
        return sentence_embeddings["data"][0]["embedding"]

    @staticmethod
    def dimension():
        return 1536
