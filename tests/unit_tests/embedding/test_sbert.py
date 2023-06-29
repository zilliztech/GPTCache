from gptcache.adapter.api import _get_model
from gptcache.embedding import SBERT


def test_sbert():
    t = SBERT("all-MiniLM-L6-v2")
    dimension = t.dimension
    data = t.to_embeddings("foo")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="sbert", model_config={"model": "all-MiniLM-L6-v2"})
    dimension = t.dimension
    data = t.to_embeddings("foo")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"

    question = [
        "what is apple?",
        "what is intel?",
        "what is openai?",
    ]
    answer = ["apple", "intel", "openai"]
    for q, _ in zip(question, answer):
        data = t.to_embeddings(q)
        assert len(data) == dimension, f"{len(data)}, {t.dimension}"
