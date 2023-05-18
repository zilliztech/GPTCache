from gptcache.embedding import PaddleNLP
from gptcache.adapter.api import _get_model
from gptcache.utils.dependency_control import prompt_install

def test_paddlenlp():
    t = PaddleNLP("ernie-3.0-nano-zh")
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
    prompt_install("protobuf==4.23.1")
    t = _get_model(model_src="paddlenlp", model_config={"model": "ernie-3.0-nano-zh"})
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
