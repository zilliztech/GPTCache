def last_content(data, **kwargs):
    return data.get("messages")[-1]["content"]


def all_content(data, **kwargs):
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            s += message["content"]
        else:
            s += message["content"] + "\n"
    return s


def nop(data, **kwargs):
    return data


def get_prompt(data, **kwargs):
    return data.get("prompt")
