from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Any, List, Union, Dict

import numpy as np


class DataType(IntEnum):
    STR = 0
    IMAGE_BASE64 = 1
    IMAGE_URL = 2


@dataclass
class Answer:
    """
    data_type:
        0: str
        1: base64 image
    """

    answer: Any
    answer_type: int = DataType.STR


@dataclass
class QuestionDep:
    """
    QuestionDep
    """

    name: str
    data: str
    dep_type: int = DataType.STR

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(
            name=d["name"],
            data=d["data"],
            dep_type=d["dep_type"]
        )


@dataclass
class Question:
    """
    Question
    """

    content: str
    deps: Optional[List[QuestionDep]] = None

    @classmethod
    def from_dict(cls, d: Dict):
        deps = []
        for dep in d["deps"]:
            deps.append(QuestionDep.from_dict(dep))
        return cls(d["content"], deps)


@dataclass
class CacheData:
    """
    CacheData
    """

    question: Union[str, Question]
    answers: List[Answer]
    embedding_data: Optional[np.ndarray] = None

    def __init__(self, question, answers, embedding_data = None):
        self.question = question
        self.answers = []
        if isinstance(answers, (str, Answer)):
            answers = [answers]
        for data in answers:
            if isinstance(data, (list, tuple)):
                self.answers.append(Answer(*data))
            elif isinstance(data, Answer):
                self.answers.append(data)
            else:
                self.answers.append(Answer(answer=data))
        self.embedding_data = embedding_data


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def batch_insert(self, all_data: List[CacheData]):
        pass

    @abstractmethod
    def get_data_by_id(self, key):
        pass

    @abstractmethod
    def mark_deleted(self, keys):
        pass

    @abstractmethod
    def clear_deleted_data(self):
        pass

    @abstractmethod
    def get_ids(self, deleted=True):
        pass

    @abstractmethod
    def count(self):
        pass

    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass
