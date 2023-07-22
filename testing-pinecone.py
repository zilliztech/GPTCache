import time
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import pdb

def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

print("Cache loading.....")

onnx = Onnx()

### you can uncomment the following lines according to which database you want to use
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("pinecone", \
                                dimension=onnx.dimension, index_name='caching', api_key='e0c287dd-b4a3-4600-ad42-5bf792decf19',\
                                    environment = 'asia-southeast1-gcp-free', metric='euclidean'))

# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("chromadb", dimension=onnx.dimension))
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("docarray", dimension=onnx.dimension))

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

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
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'user',
                    'content': question
                }
            ],
            metadata=metadata
        )
        print(f'Question: {question}')
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f'Answer: {response_text(response)}\n')
        # print("usage: ",response['usage'])