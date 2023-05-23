import time
import torch
from transformers import pipeline
from gptcache.processor.pre import get_inputs
from gptcache.manager import manager_factory
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.adapter.dolly import Dolly


def dolly_base_usage():
    onnx = Onnx()
    m = manager_factory("sqlite,faiss,local", data_dir="./dolly", vector_params={"dimension": onnx.dimension})
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_inputs,
        data_manager=m,
        embedding_func=onnx.to_embeddings
    )    

    llm = Dolly.from_model(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device=0)

    context = """George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman,
and Founding Father who served as the first president of the United States from 1789 to 1797."""
    
    for _ in range(2):
        start_time = time.time()
        answer = llm(context, cache_obj=llm_cache)
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Received: {answer[0]['generated_text']}")
        print(f"Hit cache: {answer[0].get('gptcache', False)}")


def dolly_from_hugggingface():
    onnx = Onnx()
    m = manager_factory("sqlite,faiss,local", data_dir="./dolly_hg", vector_params={"dimension": onnx.dimension})
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_inputs,
        data_manager=m,
        embedding_func=onnx.to_embeddings
    )    

    pipe = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                    trust_remote_code=True, device=0, return_full_text=True)
    llm = Dolly(pipe)

    context = """George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman,
and Founding Father who served as the first president of the United States from 1789 to 1797."""
    
    for _ in range(2):
        start_time = time.time()
        answer = llm(context, cache_obj=llm_cache)
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Received: {answer[0]['generated_text']}")
        print(f"Hit cache: {answer[0].get('gptcache', False)}")            


if __name__ == '__main__':
    dolly_base_usage()
    dolly_from_hugggingface()
