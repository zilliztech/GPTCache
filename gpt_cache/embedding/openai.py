import openai


def to_embeddings(data, **kwargs):
    messages = data.get("messages")[-1]["content"]
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=messages
    )
    return sentence_embeddings["data"][0]["embedding"]
