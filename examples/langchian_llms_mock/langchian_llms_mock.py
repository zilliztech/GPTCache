import os

from gptcache.adapter.langchain_llms import LangChainLLMs
from langchain.llms import OpenAI
from gptcache.core import cache, Config
from gptcache.cache.factory import get_ss_data_manager
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation
from gptcache.processor.post import nop as postnop
from gptcache.processor.pre import nop as prenop
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    sqlite_file = "gptcache.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    data_manager = get_ss_data_manager("sqlite", "faiss",
                                       dimension=d, max_size=8, clean_size=2, top_k=3)
    cache.init(embedding_func=mock_embeddings,
               pre_embedding_func=prenop,
               post_process_messages_func=postnop,
               data_manager=data_manager,
               evaluation_func=SearchDistanceEvaluation(),
               config=Config(
                       similarity_threshold=0,
                   ),
               )

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ]
    if not has_data:
        for i in range(10):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            cache.data_manager.save(question, answer, cache.embedding_func(question))


    llm = LangChainLLMs(OpenAI(model_name="text-ada-001", n=2, best_of=2))
    answer = llm(mock_messages)
    print(answer)


if __name__ == '__main__':
    run()
