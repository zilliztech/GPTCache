import re
from typing import Dict, Any


def last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("messages")[-1]["content"]


def last_content_without_prompt(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    last_content_str = data.get("messages")[-1]["content"]
    prompts = params.get("prompts", [])
    if prompts is None:
        return last_content_str
    pattern = "|".join(prompts)
    new_content_str = re.sub(pattern, "", last_content_str)
    return new_content_str


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


def get_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    return data.get("file").name


def get_file_bytes(data: Dict[str, Any], **_: Dict[str, Any]) -> bytes:
    return data.get("file").peek()


def get_input_str(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]


def get_input_image_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    return input_data["image"].name
