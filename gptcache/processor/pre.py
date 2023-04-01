def last_content(data, **kwargs):
    return data.get("messages")[-1]["content"]