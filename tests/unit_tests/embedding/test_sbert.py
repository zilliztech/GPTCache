from gptcache.embedding import SBERT


def test_sbert():
    t = SBERT("paraphrase-albert-small-v2")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
