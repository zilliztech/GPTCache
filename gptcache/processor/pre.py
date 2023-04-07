def last_content(data, **_):
    return data.get("messages")[-1]["content"]


def all_content(data, **_):
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            s += message["content"]
        else:
            s += message["content"] + "\n"
    return s


def nop(data, **_):
    return data


def get_prompt(data, **_):
    return data.get("prompt")
