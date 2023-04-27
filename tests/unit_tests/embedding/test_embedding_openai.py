import os
from unittest.mock import patch

from gptcache.embedding import OpenAI
from gptcache.adapter.api import _get_model


def test_embedding():
    os.environ["OPENAI_API_KEY"] = "API"

    def get_return_value(d):
        return {
          "object": "list",
          "data": [
            {
              "object": "embedding",
              "embedding": [0] * d,
              "index": 0
            }
          ],
          "model": "text-embedding-ada-002",
          "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
          }
        }

    with patch("openai.Embedding.create") as mock_create:
        dimension = 1536
        mock_create.return_value = get_return_value(dimension)
        oa = OpenAI()
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("openai.Embedding.create") as mock_create:
        dimension = 1536
        mock_create.return_value = get_return_value(dimension)
        oa = OpenAI(api_key="openai")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("openai.Embedding.create") as mock_create:
        dimension = 512
        mock_create.return_value = get_return_value(dimension)
        oa = OpenAI(model="test_embedding")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("openai.Embedding.create") as mock_create:
        dimension = 1536
        mock_create.return_value = get_return_value(dimension)
        oa = _get_model("openai")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension


