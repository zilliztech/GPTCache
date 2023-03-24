import json
import time

from scenario_cache.view import openai
from scenario_cache.core import cache
from scenario_cache.cache.data_manager import SFDataManager
from scenario_cache.similarity_evaluation.faiss import faiss_evaluation
from scenario_cache.embedding.towhee import to_embeddings as towhee_embedding

d = 768


def run():
    with open('mock_data.json', 'r') as mock_file:
        mock_data = json.load(mock_file)

    cache.init(embedding_func=towhee_embedding,
               data_manager=SFDataManager("sqlite.db", "faiss.index", d),
               evaluation_func=faiss_evaluation,
               similarity_threshold=50,
               similarity_positive=False)

    i = 0
    for pair in mock_data:
        pair["id"] = str(i)
        i += 1

    # you should OPEN it if you FIRST run it
    # print("insert data")
    # for pair in mock_data:
    #     source_messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": pair["origin"]}
    #     ]
    #     cache.data_manager.save(pair["id"], cache.embedding_func({"messages": source_messages}))
    # print("end insert data")

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
            # print("res_text", res_text)
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

    cache.data_manager.close()


if __name__ == '__main__':
    run()
