import asyncio

from gptcache.utils import import_httpx

import_httpx()

import httpx  # pylint: disable=C0413


class Client:
    """GPTCache client to send requests to GPTCache server.

    :param uri: the uri leads to the server, defaults to "http://localhost:8000".
    :type uri: str

    Example:
        .. code-block:: python

            from gptcache import client

            client = Client(uri="http://localhost:8000")
            client.put("Hi", "Hi back")
            ans = client.get("Hi")
    """

    def __init__(self, uri: str = "http://localhost:8000"):
        self._uri = uri

    async def _put(self, question: str, answer: str):
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            params = {"prompt": question}
            data = answer

            response = await client.put(self._uri, params=params, headers=headers, data=data)

        return response.status_code

    async def _get(self, question: str):
        async with httpx.AsyncClient() as client:
            params = {"prompt": question}
            response = await client.get(self._uri, params=params)

        return response.json()

    def put(self, question: str, answer: str):
        """
        :param question: the question to be put.
        :type question: str
        :param answer: the answer to the question to be put.
        :type answer: str
        :return: status code.
        """
        return asyncio.run(self._put(question, answer))

    def get(self, question: str):
        """
        :param question: the question to get an answer.
        :type question: str
        :return: answer to the question.
        """
        return asyncio.run(self._get(question))
