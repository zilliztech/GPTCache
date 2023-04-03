import json
import os
import time

from gptcache.adapter import openai
from gptcache.core import cache, Config
from gptcache.cache.factory import get_si_data_manager
from gptcache.ranker import Towhee as EvaluationTowhee
from gptcache.encoder import Towhee as EmbeddingTowhee
from gptcache.ranker.simple import pair_evaluation


def run():
    with open('mock_data.json', 'r') as mock_file:
        mock_data = json.load(mock_file)

    embedding_towhee = EmbeddingTowhee()

    # if you want more accurate results,
    # you can use towhee's results to evaluate the model,
    # it will make the results more accurate, but the cache hit rate will decrease

    # evaluation_towhee = EvaluationTowhee()
    # def sf_evaluation(src_dict, cache_dict, **kwargs):
    #     rank1 = pair_evaluation(src_dict, cache_dict, **kwargs)
    #     return rank1
    #     if rank1 <= 0.5:
    #         rank2 = evaluation_towhee.evaluation(src_dict, cache_dict, **kwargs)
    #         return rank2 if rank2 != 0 else 1
    #     return 0

    def sf_evaluation(src_dict, cache_dict, **kwargs):
        return pair_evaluation(src_dict, cache_dict, **kwargs)

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    data_manager = get_si_data_manager("sqlite", "faiss", dimension=embedding_towhee.dimension(), max_size=100000)
    cache.init(embedding_func=embedding_towhee.to_embeddings,
               data_manager=data_manager,
               evaluation_func=sf_evaluation,
               config=Config(
                   similarity_threshold=0.5,
                   similarity_positive=True,
                    ),
               )

    i = 0
    for pair in mock_data:
        pair["id"] = str(i)
        i += 1

    if not has_data:
        print("insert data")
        id_origin = {}
        for pair in mock_data:
            question = pair["origin"]
            answer = pair["id"]
            id_origin[answer] = question
            cache.data_manager.save(question, answer, cache.embedding_func(question))
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
