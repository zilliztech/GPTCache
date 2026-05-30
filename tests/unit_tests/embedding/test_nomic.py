# import os
# import types
# from unittest.mock import patch
# from gptcache.utils import import_nomic
# from gptcache.embedding import Nomic
# from gptcache.adapter.api import _get_model

# import_nomic()

# def test_nomic():
#     t = Nomic(model='nomic-embed-text-v1.5', api_key=os.getenv("NOMIC_API_KEY"), dimensionality=64)
#     data = t.to_embeddings("foo")
#     assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

#     t = _get_model(model_src="nomic", model_config={"model": "nomic-embed-text-v1.5"})
#     data = t.to_embeddings("foo")
#     assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"