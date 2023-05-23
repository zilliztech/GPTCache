import time

from gptcache.adapter.llama_cpp import Llama
from gptcache.manager import manager_factory
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.processor.pre import get_prompt


def llama_cpp_base_usage():
    onnx = Onnx()
    m = manager_factory("sqlite,faiss,local", data_dir="./llamacpp_basic", vector_params={"dimension": onnx.dimension})
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
        data_manager=m,
        embedding_func=onnx.to_embeddings
    )
    llm = Llama("./ggml-model-q4_0.bin")
    for _ in range(2):
        start_time = time.time()
        answer = llm(prompt="Q: Name the planets in the solar system? A: ", stop=["Q:", "\n"], cache_obj=llm_cache)
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Received: {answer['choices'][0]['text']}")
        print(f"Hit cache: {answer.get('gptcache', False)}")


def llama_cpp_stream_usage():
    onnx = Onnx()
    m = manager_factory("sqlite,faiss,local", data_dir="./llamacpp_stream", vector_params={"dimension": onnx.dimension})
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
        data_manager=m,
        embedding_func=onnx.to_embeddings
    )
    llm = Llama("./ggml-model-q4_0.bin")
    for _ in range(2):
        start_time = time.time()
        ret = llm(prompt="Q: Name the planets in the solar system? A: ", stop=["Q:", "\n"], stream=True, cache_obj=llm_cache)
        answer = ''
        for chunk in ret:
            answer += chunk['choices'][0]['text']
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Received: {answer}")


if __name__ == "__main__":
    llama_cpp_base_usage()
    llama_cpp_stream_usage()
    
