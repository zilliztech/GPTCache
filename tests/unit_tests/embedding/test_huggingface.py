from gptcache.embedding import Huggingface
from gptcache.adapter.api import _get_model


def test_huggingface():
    t = Huggingface("sentence-transformers/paraphrase-albert-small-v2")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="huggingface", model_config={"model": "sentence-transformers/paraphrase-albert-small-v2"})
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
