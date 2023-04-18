from gptcache.adapter.api import put, get
from gptcache.manager import CacheBase, VectorBase
from gptcache.processor.pre import get_prompt
from gptcache.processor.post import nop
from gptcache import cache, get_data_manager, Config, Cache
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation


def run_basic():
    cache.init(pre_embedding_func=get_prompt)
    put("hello", "foo")
    print(get("hello"))
    # output: foo


def run_similar_match():
    embedding_onnx = EmbeddingOnnx()
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension, top_k=10)
    data_manager = get_data_manager(cache_base, vector_base)

    evaluation = SearchDistanceEvaluation()
    inner_cache = Cache()
    inner_cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=nop,
        config=Config(similarity_threshold=0),
    )
    put("hello1", "foo1", cache_obj=inner_cache)
    put("hello2", "foo2", cache_obj=inner_cache)
    put("hello3", "foo3", cache_obj=inner_cache)

    messages = get("hello", cache_obj=inner_cache, top_k=3)
    print(messages)
    # output: ['foo1', 'foo2', 'foo3']


if __name__ == "__main__":
    run_basic()
    run_similar_match()
