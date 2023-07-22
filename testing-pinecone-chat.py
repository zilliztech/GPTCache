import time
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import pdb
from gptcache.adapter.langchain_models import LangChainChat
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

print("Cache loading.....")

def get_msg(data, **_):
    return data.get("messages")[-1].content

onnx = Onnx()
### you can uncomment the following lines according to which database you want to use
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("pinecone", \
#                                 dimension=onnx.dimension, index_name='caching', api_key='e0c287dd-b4a3-4600-ad42-5bf792decf19',\
#                                     environment = 'asia-southeast1-gcp-free', metric='euclidean'))

data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("chromadb", dimension=onnx.dimension))
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("docarray", dimension=onnx.dimension))

cache.init(
    pre_embedding_func=get_msg,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

chat = LangChainChat(chat=ChatOpenAI(temperature=0.0))

questions = [
    "tell me something about chatgpt",
    "what is chatgpt?",
]

metadata = {
    'account_id': '-123',
    'pipeline': 'completion'
}

if __name__=="__main__":
    # pdb.set_trace()
    for question in questions:
        start_time = time.time()
        messages = [HumanMessage(content=question)]
        print(chat(messages, metadata=metadata))
        print(f'Question: {question}')
        print("Time consuming: {:.2f}s".format(time.time() - start_time))