import numpy as np
import pytest
import redis

from gptcache.embedding import Onnx
from gptcache.manager import VectorBase
from gptcache.manager.vector_data.base import VectorData


def _is_valkey() -> bool:
    try:
        r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
        info = r.info("server")
        # Valkey exposes server_name: valkey and valkey_version
        return info.get("server_name") == "valkey" or "valkey_version" in info
    except Exception:
        return False


def extract_score(item):
    assert isinstance(item, (list, tuple)) and len(item) >= 2, f"Unexpected item shape: {item}"
    return float(item[0])


def is_nondecreasing(seq, *, tol=1e-9):
    return all(seq[i] <= seq[i + 1] + tol for i in range(len(seq) - 1))


def _is_nonincreasing(seq, tol=1e-9):
    return all(seq[i] >= seq[i + 1] - tol for i in range(len(seq) - 1))


@pytest.fixture(autouse=True)
def clean_redis_db():
    r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
    try:
        r.flushdb()
    except Exception:
        pass
    yield
    try:
        r.flushdb()
    except Exception:
        pass


def test_redis_vector_store():
    encoder = Onnx()
    dim = encoder.dimension
    vector_base = VectorBase("redis", dimension=dim)
    vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(10)])

    search_res = vector_base.search(np.random.rand(dim))
    print(search_res)
    assert len(search_res) == 1

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 10

    vector_base.delete([i for i in range(5)])

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 5


@pytest.mark.parametrize("ascending", [True])
def test_redis_vector_store_sortby_supported(monkeypatch, ascending):
    """
    Force the probe to say SORTBY is supported, ensuring the query adds `.sort_by("score")`.
    With your return shape (score,id), verify scores are ascending (non-decreasing).
    """
    encoder = Onnx()
    dim = encoder.dimension
    vector_base = VectorBase("redis", dimension=dim)
    vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(30)])

    from gptcache.manager.vector_data.redis_vectorstore import RedisVectorStore

    def _always_supports_sortby(self, index_name, field) -> bool:  # noqa: ARG002
        return not _is_valkey()

    monkeypatch.setattr(
        RedisVectorStore,
        "_check_sortby_support",
        _always_supports_sortby,
        raising=True,
    )

    k = 10
    query_vec = np.random.rand(dim)
    res = vector_base.search(query_vec, top_k=k)
    assert isinstance(res, (list, tuple))
    assert len(res) == k

    scores = [extract_score(item) for item in res]
    assert is_nondecreasing(scores), f"Scores not sorted ASC: {scores}"


def test_redis_vector_store_sortby_unsupported(monkeypatch):
    """
    Force the probe to say SORTBY is NOT supported (e.g., Valkey without SORTBY).
    The search should still succeed and return k results;
    """
    encoder = Onnx()
    dim = encoder.dimension
    vector_base = VectorBase("redis", dimension=dim)
    vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(20)])

    from gptcache.manager.vector_data.redis_vectorstore import RedisVectorStore

    def _never_supports_sortby(self, index_name, field):  # noqa: ARG002
        return False

    monkeypatch.setattr(
        RedisVectorStore,
        "_check_sortby_support",
        _never_supports_sortby,
        raising=True,
    )

    k = 10
    query_vec = np.random.rand(dim)
    res = vector_base.search(query_vec, top_k=k)
    assert isinstance(res, (list, tuple))
    assert len(res) == k

    scores = [extract_score(item) for item in res]
    if _is_valkey():
        assert is_nondecreasing(scores), f"Scores not sorted ASC: {scores}"
    else:
        assert not is_nondecreasing(scores) and not _is_nonincreasing(scores), f"Scores sorted: {scores}"
