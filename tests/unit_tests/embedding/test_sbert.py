import pytest

from gptcache.adapter.api import _get_model
from gptcache.embedding import SBERT


@pytest.mark.tags("L2")
def test_sbert():
    t = SBERT("all-MiniLM-L6-v2")
    dimension = t.dimension
    data = t.to_embeddings("foo")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="sbert", model_config={"model": "all-MiniLM-L6-v2"})
    dimension = t.dimension
    data = t.to_embeddings("foo")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"