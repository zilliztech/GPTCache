from gptcache.embedding import Onnx 


def test_onnx():
    t = Onnx()
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
