import re
import string
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


def _get_pattern_value(pattern_str: str, value_str: str):
    literal_text_arr = []
    field_name_arr = []
    for literal_text, field_name, _, _ in string.Formatter().parse(pattern_str):
        literal_text_arr.append(literal_text)
        if field_name is not None:
            field_name_arr.append(
                field_name if field_name else str(len(field_name_arr))
            )

    pattern_values = {}
    last_end = 0
    for i, (literal_text, field_name) in enumerate(zip(literal_text_arr, field_name_arr)):
        start = value_str.find(literal_text, last_end)
        if i == len(literal_text_arr) - 1:
            end = len(value_str)
        else:
            end = value_str.find(literal_text_arr[i + 1], start + 1)
        if start == -1 or end == -1:
            break
        start += len(literal_text)
        pattern_values[field_name] = value_str[start:end]
        last_end = end
    return pattern_values


def last_content_without_template(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    """get the last content's template values of the message list without template content.

    When considering a cache agent or chain, the majority of the content consists of template content,
    while the essential information is simply a list of parameters within the template.
    In this way, the cache key is composed of a string made up of all the parameter values in the list.

    WARNING: Two parameters without intervals cannot appear in the template,
    for example: template = "{foo}{hoo}" is not supported,
    but template = "{foo}:{hoo}" is supported

    :param data: the user llm request data
    :type data: Dict[str, Any]

    :Example with str template:
        .. code-block:: python

            from gptcache import Config
            from gptcache.processor.pre import last_content_without_template

            template_obj = "tell me a joke about {subject}"
            prompt = template_obj.format(subject="animal")
            value = last_content_without_template(
                data={"messages": [{"content": prompt}]}, cache_config=Config(template=template_obj)
            )
            print(value)
            # ['animal']

    :Example with langchain template:
        .. code-block:: python

            from langchain import PromptTemplate

            from gptcache import Config
            from gptcache.processor.pre import last_content_without_template

            template_obj = PromptTemplate.from_template("tell me a joke about {subject}")
            prompt = template_obj.format(subject="animal")

            value = last_content_without_template(
                data={"messages": [{"content": prompt}]},
                cache_config=Config(template=template_obj.template),
            )
            print(value)
            # ['animal']

    NOTE: At present, only the simple PromptTemplate in langchain is supported.
    For ChatPromptTemplate, it needs to be adjusted according to the template array.
    If you need to use it, you need to pass in the final dialog template yourself.
    The reason why it cannot be advanced is that ChatPromptTemplate
    does not provide a method to directly return the template string.
    """
    last_content_str = data.get("messages")[-1]["content"]
    cache_config = params.get("cache_config", None)
    if not (cache_config and cache_config.template):
        return last_content_str

    pattern_value = _get_pattern_value(cache_config.template, last_content_str)
    return str(list(pattern_value.values()))


def all_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    """get all content of the message list

    :param data: the user llm request data
    :type data: Dict[str, Any]

    :Example:
        .. code-block:: python

            from gptcache.processor.pre import all_content

            content = all_content(
                {"messages": [{"content": "foo1"}, {"content": "foo2"}]}
            )
            # content = "foo1\\nfoo2"
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


def get_messages_last_content(data: Dict[str, Any], **_: Any) -> str:
    """ get the last content of the llm request messages array

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import get_messages_last_content

            content = get_messages_last_content({"messages": [{"content": "hello"}, {"content": "world"}]})
            # "world"
    """
    return data.get("messages")[-1].content


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


def concat_all_queries(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    """

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from gptcache.processor.pre import concat_all_queries

            content = concat_all_queries({"messages": [{"role": "system", "content": "hello"},
                {"role": "user", "content": "world"},
                {"role": "assistant", "content": "alice"}]})

    """
    cache_config = params.get("cache_config", None)
    skip_list = cache_config.skip_list
    context_len = cache_config.context_len
    context_len = context_len * 2
    s = ""
    messages = data.get("messages")
    length = min(context_len, len(messages))
    messages = messages[len(messages) - length:]
    for i, message in enumerate(messages):
        if message["role"] in skip_list:
            continue
        if i == len(messages) - 1:
            s += f'{message["role"].upper()}: {message["content"]}'
        else:
            s += f'{message["role"].upper()}: {message["content"]}\n'
    return s
# ── New pre-processors ────────────────────────────────────────────────────────
 
def get_content_by_role(data: Dict[str, Any], **params: Dict[str, Any]) -> str:
    """Get concatenated content of messages that match a specific role.
 
    Useful when you only want to cache the user-side of a conversation,
    ignoring system prompts or assistant turns entirely.
 
    :param data: the user llm request data
    :type data: Dict[str, Any]
    :param params: must contain a ``role`` key (``"user"``, ``"system"``, or ``"assistant"``)
    :type params: Dict[str, Any]
 
    Example:
        .. code-block:: python
 
            from gptcache.processor.pre import get_content_by_role
 
            content = get_content_by_role(
                {
                    "messages": [
                        {"role": "system",    "content": "You are a helpful assistant."},
                        {"role": "user",      "content": "Who won the world series in 2020?"},
                        {"role": "assistant", "content": "The Los Angeles Dodgers won."},
                        {"role": "user",      "content": "Where was it played?"},
                    ]
                },
                role="user",
            )
            # content = "Who won the world series in 2020?\nWhere was it played?"
    """
    role = params.get("role", "user")
    messages = data.get("messages", [])
    parts = [m["content"] for m in messages if m.get("role") == role]
    return "\n".join(parts)
 
 
def get_system_prompt_and_last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """Return the system prompt (if present) concatenated with the last user message.
 
    Many LLM applications keep the same system prompt while varying only the user
    turn.  Including the system prompt in the cache key prevents false cache hits
    when different system prompts are paired with the same user query.
 
    :param data: the user llm request data
    :type data: Dict[str, Any]
 
    Example:
        .. code-block:: python
 
            from gptcache.processor.pre import get_system_prompt_and_last_content
 
            content = get_system_prompt_and_last_content(
                {
                    "messages": [
                        {"role": "system",    "content": "You are a helpful assistant."},
                        {"role": "user",      "content": "Who won the world series in 2020?"},
                        {"role": "assistant", "content": "The Los Angeles Dodgers won."},
                        {"role": "user",      "content": "Where was it played?"},
                    ]
                }
            )
            # content = "You are a helpful assistant.\nWhere was it played?"
    """
    messages = data.get("messages", [])
    system_content = next(
        (m["content"] for m in messages if m.get("role") == "system"), ""
    )
    last = messages[-1]["content"] if messages else ""
    return f"{system_content}\n{last}" if system_content else last
 
 
def last_n_turns_content(data: Dict[str, Any], **params: Dict[str, Any]) -> str:
    """Return the concatenated content of the last *n* conversational turns.
 
    A "turn" is one user message plus its following assistant reply (2 messages).
    System messages are always excluded.  This gives the cache a short but
    meaningful context window without exploding the key space.
 
    :param data: the user llm request data
    :type data: Dict[str, Any]
    :param params: accepts ``n`` (int, default ``2``) – number of turns to include
    :type params: Dict[str, Any]
 
    Example:
        .. code-block:: python
 
            from gptcache.processor.pre import last_n_turns_content
 
            content = last_n_turns_content(
                {
                    "messages": [
                        {"role": "system",    "content": "You are a helpful assistant."},
                        {"role": "user",      "content": "Who won the world series in 2020?"},
                        {"role": "assistant", "content": "The Los Angeles Dodgers won."},
                        {"role": "user",      "content": "Where was it played?"},
                    ]
                },
                n=1,
            )
            # content = "ASSISTANT: The Los Angeles Dodgers won.\nUSER: Where was it played?"
    """
    n = int(params.get("n", 2))
    messages = [m for m in data.get("messages", []) if m.get("role") != "system"]
    window = messages[-(n * 2):]
    return "\n".join(f'{m["role"].upper()}: {m["content"]}' for m in window)
 
 
def get_model_and_last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    """Prefix the last message content with the requested model name.
 
    When a single GPTCache instance serves multiple models (e.g. gpt-3.5-turbo
    and gpt-4), the same question should NOT share a cached response across
    models.  Prepending the model name creates isolated key spaces automatically.
 
    :param data: the user llm request data, expected to contain a top-level
                 ``"model"`` key alongside ``"messages"``
    :type data: Dict[str, Any]
 
    Example:
        .. code-block:: python
 
            from gptcache.processor.pre import get_model_and_last_content
 
            content = get_model_and_last_content(
                {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "user", "content": "Where was it played?"},
                    ],
                }
            )
            # content = "gpt-4::Where was it played?"
    """
    model = data.get("model", "")
    messages = data.get("messages", [])
    last = messages[-1]["content"] if messages else ""
    return f"{model}::{last}"

