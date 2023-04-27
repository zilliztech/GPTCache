from gptcache.embedding import Onnx
from gptcache.adapter.api import _get_model


def test_onnx():
    t = Onnx()
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    t = _get_model("onnx")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
