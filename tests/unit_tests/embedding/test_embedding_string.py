from gptcache.embedding.string import to_embeddings


def test_embedding():
    message = to_embeddings('foo')
    assert message == 'foo'
