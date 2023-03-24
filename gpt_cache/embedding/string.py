def to_embeddings(data, **kwargs):
    messages = data.get("messages")
    return messages[-1]["content"]
