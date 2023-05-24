from gptcache.adapter.api import _get_model
from gptcache.embedding import Rwkv


def test_rwkv():
    t = Rwkv("sgugger/rwkv-430M-pile")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="rwkv", model_config={"model": "sgugger/rwkv-430M-pile"})
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

