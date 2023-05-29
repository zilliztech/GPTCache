from typing import Any, Dict

from gptcache.processor import ContextProcess
from gptcache.utils import import_selective_context

import_selective_context()

from selective_context import SelectiveContext  # pylint: disable=C0413


class SelectiveContextProcess(ContextProcess):
    """A context processor for selecting context

    Need to download the corresponding model before use, the default English model is: en_core_web_sm

    `pip install spacy && python -m spacy download en_core_web_sm`

    :param model_type: the selective context model name, default value is 'gpt2'
    :type model_type: str
    :param lang: the content lang type, default value is 'en'.
    :type lang: str
    :param reduce_ratio: selective context ratio. The range for the value is between 0 and 1, with a default value of 0.35.
    :type reduce_ratio: float
    :param reduce_level: selective context level. The valid values include 'sent', 'phrase', and 'token', with the default value being 'phrase'.
    :type reduce_level: str

    more details: https://github.com/liyucheng09/Selective_Context

    Example:
        .. code-block:: python

            from gptcache.processor.context.selective_context import SelectiveContextProcess

            context_process = SelectiveContextProcess()
            cache.init(pre_embedding_func=context_process.pre_process)
    """

    content: str = ""

    def __init__(
            self,
            model_type: str = "gpt2",
            lang: str = "en",
            reduce_ratio: float = 0.35,
            reduce_level: str = "phrase",
    ):
        self.sc = SelectiveContext(model_type=model_type, lang=lang)
        self.reduce_ratio = reduce_ratio
        self.reduce_level = reduce_level

    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        for query in data["messages"]:
            self.content += f"{query['role']}: {query['content']} \n"

    def process_all_content(self) -> (Any, Any):
        selective_content, _ = self.sc(
            self.content, reduce_ratio=self.reduce_ratio, reduce_level=self.reduce_level
        )
        return self.content, selective_content
