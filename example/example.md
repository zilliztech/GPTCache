# Example

## [Basic example](map/map_manager.py)

How to use the map to cache data.

## [Sqlite + Faiss manage cache data](sf_mock/sf_manager.py)

Before running this case, you should install the `faiss-cpu`.

```bash
pip install faiss-cpu
```

How to use the [sqlite](https://www.sqlite.org/index.html) to store the scale data and the faiss to query the vector data.

## [Sqlite + Faiss + Towhee](sf_towhee/sf_manager.py)

Before running this case, you should install the `faiss-cpu` and `towhee`.

```bash
pip install faiss-cpu
pip install towhee==0.9.0
```

On the basis of the above example, use [towhee](https://towhee.io/) for embedding operation

Note: the default embedding model only support the **ENGLISH**. If you want to use the Chinese, you can use the `uer/albert-base-chinese-cluecorpussmall` model. For other languages, you should use the corresponding model.

## [Sqlite + Milvus + Towhee](sqlite_milvus_mock/sqlite_milvus_mock.py)

Before running this case, you should install the `faiss-cpu`, `towhee` and `pymilvus`.

```bash
pip install faiss-cpu
pip install towhee==0.9.0
pip install pymilvus
```

How to use the [sqlite](https://www.sqlite.org/index.html) to store the scale data and the [milvus](https://milvus.io/docs) to store the vector data.

## [Benchmark](benchmark/benchmark_sf_towhee.py)

Before running this case, you should install the `faiss-cpu` and `towhee`.

```bash
pip install faiss-cpu
pip install towhee==0.9.0
```

The benchmark script about the `Sqlite + Faiss + Towhee`

[Test data source](benchmark/mock_data.json): Randomly scrape some information from the webpage (origin), and then let chatgpt produce corresponding data (similar).

- **threshold**: answer evaluation threshold, A smaller value means higher consistency with the content in the cache, a lower cache hit rate, and a lower cache miss hit; a larger value means higher tolerance, a higher cache hit rate, and at the same time also have higher cache misses.
- **positive**: effective cache hit, which means entering `similar` to search and get the same result as `origin`
- **negative**: cache hit but the result is wrong, which means entering `similar` to search and get the different result as `origin`
- **fail count**: cache miss

data file: [mock_data.json](benchmark/mock_data.json)
similarity evaluation func: pair_evaluation (search distance)

 | threshold | average time | positive | negative | fail count |
|-----------|--------------|----------|----------|------------|
| 20        | 0.04s        | 455      | 27       | 517        |
| 50        | 0.09s        | 871      | 86       | 42         |
| 100       | 0.12s        | 905      | 93       | 1          |
