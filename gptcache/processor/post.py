import random
from typing import List, Any

import numpy

from gptcache.utils import softmax


def random_one(messages: List[Any]) -> Any:
    """Randomly select one result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import random_one

            messages = ["message 1", "message 2", "message 3"]
            answer = random_one(messages)
    """
    return random.choice(messages)


def first(messages: List[Any]) -> Any:
    """Get the first result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import first

            messages = ["message 1", "message 2", "message 3"]
            answer = first(messages)
            assert answer = messages[0]
    """
    return messages[0]


def nop(messages: List[Any]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer = messages
    """
    return messages


def temperature_softmax(messages: List[Any], scores: List[float], temperature: float = 0.0) -> Any:
    """Post processing with temperature softmax after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to `messages`
    :type scores: List[float]
    :param temperature: A non-negative number of sampling temperature, defaults to 0.
                        A higher temperature makes the output more random.
                        A lower temperature means a more deterministic and confident output.
    :type temperature: float

    Example:
        .. code-block:: python

            from gptcache.processor.post import temperature_softmax

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer = temperature_softmax(messages, scores, temperature=0.5)
    """

    if temperature > 0:
        scores = softmax([x / temperature for x in scores])
        return numpy.random.choice(messages, size=1, p=scores)[0]
    else:
        m_s = list(zip(messages, scores))
        return sorted(m_s, key=lambda x: x[1], reverse=True)[0][0]
