import random
from typing import List, Any


def random_one(messages: List[Any]) -> Any:
    return random.choice(messages)


def first(messages: List[Any]) -> Any:
    return messages[0]


def nop(messages: List[Any]) -> Any:
    return messages
