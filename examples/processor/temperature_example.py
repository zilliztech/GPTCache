import os

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx
from gptcache.processor.pre import get_prompt
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.api import put, get


# Init cache with vector store
if os.path.exists("faiss.index"):
    os.remove("faiss.index")
if os.path.exists("sqlite.db"):
    os.remove("sqlite.db")

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})


cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )
# cache.config = Config(similarity_threshold=0.2)

# Input some prepared data to mock a cache with data stored
my_data = [
    {"Q": "What is the most popular vector database?", "A": "Milvus!"},
    {"Q": "What are most popular vector databases?", "A": "Milvus, Milvus, still Milvus ..."},
    {"Q": "What is vector database?", "A": "Vector database is xxxx."},
    {"Q": "Is Milvus an open-source vector database?", "A": "Yes, Milvus is open source."},
    {"Q": "What is Zilliz cloud?", "A": "Zilliz cloud provides vector database on cloud."},
    {"Q": "What is Milvus?", "A": "Milvus is an open-source vector database."},
    {"Q": "Can you recommend a vector database?", "A": "Sure, Milvus is a good choice for vector database."},
    {"Q": "Is Zilliz Cloud free?", "A": "No, Zilliz Cloud charges for instance."},
    {"Q": "How many credits can I get for Zilliz Cloud?", "A": "A new user of Zilliz Cloud will get 350 credits."},
    {"Q": "Do you like GPTCache?", "A": "Yea! GPTCache is great!"},
]

for qa in my_data:
    put(prompt=qa["Q"], data=qa["A"], skip_cache=True)


# use cache without temperature (temperature=0.0)
for _ in range(5):
    answer = get(prompt="popular vector database")
    print(answer)

# use cache with temperature (eg. temperature=2.0)
for _ in range(5):
    answer = get(prompt="popular vector database", temperature=2.0)
    print(answer)