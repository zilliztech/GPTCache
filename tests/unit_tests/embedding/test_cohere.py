import os
import types
from unittest.mock import patch
from gptcache.utils import import_cohere
from gptcache.embedding import Cohere

import_cohere()


def test_embedding():
    os.environ["CO_API_KEY"] = "API"

    with patch("cohere.Client.embed") as mock_create:
        dimension = 4096
        mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])
        c1 = Cohere()
        assert c1.dimension == dimension
        assert len(c1.to_embeddings("foo")) == dimension

    with patch("cohere.Client.embed") as mock_create:
        dimension = 512
        mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])
        c1 = Cohere("foo")
        assert c1.dimension == dimension
        assert len(c1.to_embeddings("foo")) == dimension
