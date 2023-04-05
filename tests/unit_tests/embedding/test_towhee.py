from gptcache.embedding import Towhee


def test_towhee():
    t = Towhee()
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension(), f"{len(data)}, {t.dimension}"