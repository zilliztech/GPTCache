import re
from typing import Dict, Any


def last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    """get the last content of the message list

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import last_content

            content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
            # content = "foo2"
    """
    return data.get("messages")[-1]["content"]


def last_content_without_prompt(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    """get the last content of the message list without prompts content

    :param data: the user llm request data
    :type data: Dict[str, Any]
    :param params: the special gptcache params, like prompts param in the cache object
    :type params: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import last_content_without_prompt

            content = last_content_without_prompt(
                    {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=["foo"]
                )
            # content = "2"
    """

    last_content_str = data.get("messages")[-1]["content"]
    prompts = params.get("prompts", [])
    if prompts is None:
        return last_content_str
    pattern = "|".join(prompts)
    new_content_str = re.sub(pattern, "", last_content_str)
    return new_content_str


def all_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    """ get all content of the message list

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import all_content

            content = all_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
            # content = "foo1\nfoo2"
    """
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            s += message["content"]
        else:
            s += message["content"] + "\n"
    return s


def nop(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    """do nothing of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import nop

            content = nop({"str": "hello"})
            # {"str": "hello"}
    """
    return data


def get_prompt(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    """get the prompt of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_prompt

            content = get_prompt({"prompt": "foo"})
            # "foo"
    """
    return data.get("prompt")


def get_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """get the file name of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_file_name

            file = open("test.txt", "a")
            content = get_file_name({"file": file})
            # "test.txt"
    """
    return data.get("file").name


def get_file_bytes(data: Dict[str, Any], **_: Dict[str, Any]) -> bytes:
    """get the file bytes of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_file_bytes

            content = get_file_bytes({"file": open("test.txt", "rb")})
    """
    return data.get("file").peek()


def get_input_str(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """get the image and question str of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_input_str

            content = get_input_str({"input": {"image": open("test.png", "rb"), "question": "foo"}})
    """
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]


def get_input_image_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """get the image file name of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_input_image_file_name

            content = get_input_image_file_name({"input": {"image": open("test.png", "rb")}})
            # "test.png"
    """
    input_data = data.get("input")
    return input_data["image"].name


def get_image_question(data: Dict[str, Any], **_: Dict[str, Any]) -> str:  # pragma: no cover
    """get the image and question str of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_image_question

            content = get_image_question({"image": open("test.png", "rb"), "question": "foo"})
    """
    img = data.get("image")
    data_img = str(open(img, "rb").peek()) if isinstance(img, str) else str(img)  # pylint: disable=consider-using-with
    return data_img + data.get("question")


def get_image(data: Dict[str, Any], **_: Dict[str, Any]) -> str:  # pragma: no cover
    """get the image of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_image

            content = get_image({"image": open("test.png", "rb")})
            # "test.png"
    """
    return data.get("image")


def get_inputs(data: Dict[str, Any], **_: Dict[str, Any]):
    """get the inputs of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_inputs

            content = get_inputs({"inputs": "hello"})
            # "hello"
    """
    return data.get("inputs")


def get_openai_moderation_input(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """get the input param of the openai moderation request params

    :param data: the user openai moderation request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_openai_moderation_input

            content = get_openai_moderation_input({"input": ["hello", "world"]})
            # "['hello', 'world']"
    """

    return str(data.get("input"))
