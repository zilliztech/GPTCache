import json
import time

from gpt_cache.view import openai
from gpt_cache.core import cache
from gpt_cache.cache.factory import get_data_manager
from gpt_cache.similarity_evaluation.faiss import faiss_evaluation
from gpt_cache.embedding.towhee import Towhee


def run():
    with open('mock_data.json', 'r') as mock_file:
        mock_data = json.load(mock_file)

    towhee = Towhee()
    cache.init(embedding_func=towhee.to_embeddings,
               data_manager=get_data_manager("sqlite_faiss",  dimension=towhee.dimension()),
               evaluation_func=faiss_evaluation,
               similarity_threshold=50,
               similarity_positive=False)

    i = 0
    for pair in mock_data:
        pair["id"] = str(i)
        i += 1

    # you should OPEN it if you FIRST run it
    print("insert data")
    for pair in mock_data:
        cache.data_manager.save(pair["id"], cache.embedding_func(pair["origin"]))
    print("end insert data")

    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count = 0
    for pair in mock_data:
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]}
        ]
        try:
            start_time = time.time()
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=mock_messages,
            )
            res_text = openai.get_message_from_openai_answer(res)
            if res_text == pair["id"]:
                hit_cache_positive += 1
            else:
                hit_cache_negative += 1
            consume_time = time.time() - start_time
            all_time += consume_time
            print("cache hint time consuming: {:.2f}s".format(consume_time))
        except:
            fail_count += 1

    print("average time: {:.2f}s".format(all_time / len(mock_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count:", fail_count)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())


if __name__ == '__main__':
    run()
