# Example

Before running an example, **MUST** enter the specific example directory. For example, to run the `map` example, switch to the `map` directory, and then use python to run it.

```bash
git clone https://github.com/zilliztech/GPTCache.git
cd GPTCache

cd examples/map
python map.py
```

If the running example includes a model or complex third-party library (like: faiss, towhee), **the first run** may take **some time** as it needs to download the model runtime environment, model data, dependencies, etc. However, the subsequent runs will be significantly faster.

## [Basic example](map/map.py)

How to use the map to cache data.

## [Sqlite + Faiss manage cache data](sqlite_faiss_mock/sqlite_faiss_mock.py)

How to use the [sqlite](https://www.sqlite.org/index.html) to store the scale data and the faiss to query the vector data.

## [Sqlite + Faiss + Towhee](sqlite_faiss_towhee/sqlite_faiss_towhee.py)

On the basis of the above example, use [towhee](https://towhee.io/) for embedding operation

## [Sqlite + Milvus + Towhee](sqlite_milvus_mock/sqlite_milvus_mock.py)

How to use the [sqlite](https://www.sqlite.org/index.html) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

## [PostgreSQL + Milvus](postgresql_milvus_mock/postgresql_milvus_mock.py)

How to use the [PostgreSQL](https://www.postgresql.org/) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

> Note: you can set with your own **postgresql** with the `sql_url` parameter, such as the pseudocode `get_ss_data_manager("postgresql", "faiss", sql_url="postgresql+psycopg2://<username>:<password>@<host>:<port>/<database>")`
>
> And it is same as the **mysql**, **mariadb**, **sql server** and **oracle** database, more details refer to [the engine documentation](https://docs.sqlalchemy.org/en/20/core/engines.html#supported-databases).

## [Mysql + Milvus](mysql_milvus_mock/mysql_milvus_mock.py)

How to use the [MySQL](https://www.mysql.com/) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

## [MariaDB + Milvus](mariadb_milvus_mock/mariadb_milvus_mock.py)

How to use the [MariaDB](https://mariadb.org/) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

## [SQL Server + Milvus](mssql_milvus_mock/mssql_milvus_mock.py)

How to use the [SQL Server](https://www.microsoft.com/en-us/sql-server/) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

## [Oracle+Milvus](oracle_milvus_mock/oracle_milvus_mock.py)

How to use the [Oracle](https://www.oracle.com/) to store the scale data and the [Milvus](https://milvus.io/docs) or [Zilliz Cloud](https://cloud.zilliz.com/) to store the vector data.

## [Benchmark](benchmark/benchmark_sqlite_faiss_towhee.py)

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
