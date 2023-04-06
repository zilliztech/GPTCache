from gptcache.embedding import Huggingface


def test_huggingface():
    t = Huggingface('sentence-transformers/paraphrase-albert-small-v2')
    data = t.to_embeddings('foo')
    assert len(data) == t.dimension, f'{len(data)}, {t.dimension}'