# Example

## How to set the `embedding` function

## How to set the `data manager` class

## How to set the `similarity_evaluation` interface

## [Benchmark](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py)

The benchmark script about the `Sqlite + Faiss + ONNX`

[Test data source](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/mock_data.json): Randomly scrape some information from the webpage (origin), and then let chatgpt produce corresponding data (similar).

- **threshold**: answer evaluation threshold, A smaller value means higher consistency with the content in the cache, a lower cache hit rate, and a lower cache miss hit; a larger value means higher tolerance, a higher cache hit rate, and at the same time also have higher cache misses.
- **positive**: effective cache hit, which means entering `similar` to search and get the same result as `origin`
- **negative**: cache hit but the result is wrong, which means entering `similar` to search and get the different result as `origin`
- **fail count**: cache miss

data file: [mock_data.json](https://github.com/zilliztech/GPTCache/tree/main/examples/benchmark/mock_data.json)
similarity evaluation func: pair_evaluation (search distance)

 | threshold | average time | positive | negative | fail count |
|-----------|--------------|----------|----------|------------|
| 20        | 0.04s        | 455      | 27       | 517        |
| 50        | 0.09s        | 871      | 86       | 42         |
| 100       | 0.12s        | 905      | 93       | 1          |
