from abc import ABCMeta, abstractmethod
from typing import Dict, Any


class ContextProcess(metaclass=ABCMeta):
    """ContextProcess: the context process interfacer"""
    @abstractmethod
    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        pass

    @abstractmethod
    def process_all_content(self) -> (Any, Any):
        pass

    def pre_process(self, data: Dict[str, Any], **params: Dict[str, Any]) -> (Any, Any):
        self.format_all_content(data, **params)
        return self.process_all_content()
