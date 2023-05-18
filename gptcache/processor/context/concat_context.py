from typing import Any, Dict

from gptcache.processor import ContextProcess


class ConcatContextProcess(ContextProcess):
    """A concat context processor simply concat the context
    """

    content: str = ""

    def __init__(
            self
    ):
        self.content = ""
        self.concat_content = ""

    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        for query in data["messages"]:
            self.content += f"{query['role']}: {query['content']} \n"
            self.concat_content += query["content"]

    def process_all_content(self) -> (Any, Any):
        return self.content, self.concat_content
