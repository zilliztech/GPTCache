from typing import Dict, Any


def last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("messages")[-1]["content"]


def all_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            s += message["content"]
        else:
            s += message["content"] + "\n"
    return s


def nop(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data


def get_prompt(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("prompt")
