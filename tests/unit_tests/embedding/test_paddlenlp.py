import subprocess
from gptcache.embedding import PaddleNLP
from gptcache.adapter.api import _get_model


def test_paddlenlp():
    subprocess.check_call("export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python;", shell=True)
    
    t = PaddleNLP("ernie-3.0-nano-zh")
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
  
    t = _get_model(model_src="paddlenlp", model_config={"model": "ernie-3.0-nano-zh"})
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
