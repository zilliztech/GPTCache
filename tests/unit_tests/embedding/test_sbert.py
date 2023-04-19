from gptcache.embedding import SBERT


def test_sbert():
    t = SBERT("paraphrase-albert-small-v2")
    dimension = t.dimension
    data = t.to_embeddings("foo")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
