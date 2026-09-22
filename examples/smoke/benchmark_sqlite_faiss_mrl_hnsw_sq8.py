"""SMOKE TEST - not a benchmark.

Drives openai.ChatCompletion; the measured "time" is dominated by network
round-trips and is not meaningful for GPTCache performance work.
Use examples/benchmark/benchmark_qqp.py for memory and search-speed numbers.

Original docstring:

Benchmark: MRL-truncated SBERT + FAISS HNSW+SQ8

This benchmark measures the full MRL optimization pipeline:
  Embedding (nomic-embed-text-v1.5) → MRL Truncation (768→256) → SQ8 → HNSW

Compare results against:
  - benchmark_sqlite_faiss_onnx.py         (Flat, 768d, float32)
  - benchmark_sqlite_faiss_hnsw_sq8_onnx.py (HNSW+SQ8, 768d, uint8)
"""

import json
import os
import time

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.embedding import SBERTMRL
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


TARGET_DIM = 256


def run():
    with open("mock_data.json", "r") as mock_file:
        mock_data = json.load(mock_file)

    # MRL-enabled embedding model with truncation to TARGET_DIM
    embedding_mrl = SBERTMRL(
        model="nomic-ai/nomic-embed-text-v1.5",
        target_dim=TARGET_DIM,
    )
    print(f"Embedding model: nomic-embed-text-v1.5 (MRL truncated to {TARGET_DIM}d)")
    print(f"Reported dimension: {embedding_mrl.dimension}")

    class WrapEvaluation(SearchDistanceEvaluation):
        def evaluation(self, src_dict, cache_dict, **kwargs):
            return super().evaluation(src_dict, cache_dict, **kwargs)

        def range(self):
            return super().range()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    cache_base = CacheBase("sqlite")
    vector_base = VectorBase(
        "faiss",
        dimension=TARGET_DIM,
        index_type="hnsw_sq8",
    )
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)
    cache.init(
        embedding_func=embedding_mrl.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=WrapEvaluation(),
        config=Config(similarity_threshold=0.95),
    )

    i = 0
    for pair in mock_data:
        pair["id"] = str(i)
        i += 1

    if not has_data:
        print("insert data")
        start_time = time.time()
        questions, answers = map(
            list, zip(*((pair["origin"], pair["id"]) for pair in mock_data))
        )
        cache.import_data(questions=questions, answers=answers)
        print(
            "end insert data, time consuming: {:.2f}s".format(time.time() - start_time)
        )

    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count = 0
    for pair in mock_data:
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]},
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
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"MRL + HNSW+SQ8 Benchmark Results (dim={TARGET_DIM})")
    print("=" * 60)
    print("average time: {:.2f}s".format(all_time / len(mock_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count:", fail_count)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())

    data_manager.close()
    # --- Storage size measurement ---
    print("\n--- Storage Sizes ---")
    for filepath in [faiss_file, sqlite_file]:
        if os.path.isfile(filepath):
            size_bytes = os.path.getsize(filepath)
            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.2f} KB"
            else:
                size_str = f"{size_bytes} B"
            print(f"  {filepath}: {size_str} ({size_bytes:,} bytes)")
        else:
            print(f"  {filepath}: FILE NOT FOUND!")
    # Also check for tombstone file
    tombstone_file = faiss_file + ".tombstones.npy"
    if os.path.isfile(tombstone_file):
        size_bytes = os.path.getsize(tombstone_file)
        print(f"  {tombstone_file}: {size_bytes:,} bytes")


if __name__ == "__main__":
    run()
