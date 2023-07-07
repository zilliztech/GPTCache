from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
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
        return cls(name=d["name"], data=d["data"], dep_type=d["dep_type"])


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
    session_id: Optional[str] = None
    create_on: Optional[datetime] = None
    last_access: Optional[datetime] = None

    def __init__(
        self,
        question,
        answers,
        embedding_data=None,
        session_id=None,
        create_on=None,
        last_access=None,
    ):
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
        self.session_id = session_id
        self.create_on = create_on
        self.last_access = last_access


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
    def count(self, state: int = 0, is_all: bool = False):
        pass

    def flush(self):
        pass

    @abstractmethod
    def add_session(self, question_id, session_id, session_question):
        pass

    @abstractmethod
    def list_sessions(self, session_id, key):
        pass

    @abstractmethod
    def delete_session(self, keys):
        pass

    @abstractmethod
    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        pass

    @abstractmethod
    def close(self):
        pass
