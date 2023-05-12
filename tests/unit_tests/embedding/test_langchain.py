from gptcache.embedding import LangChain
from gptcache.utils import import_langchain

import_langchain()
from langchain.embeddings import FakeEmbeddings


def test_langchain_embedding():
    size = 10
    l = LangChain(embeddings=FakeEmbeddings(size=size))
    data = l.to_embeddings("foo")
    assert len(data) == size

    l = LangChain(embeddings=FakeEmbeddings(size=size), dimension=size)
    data = l.to_embeddings("foo")
    assert len(data) == size
