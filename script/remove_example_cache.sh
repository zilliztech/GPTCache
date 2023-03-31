#!/bin/bash

parent_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
find "$parent_dir/example" \( -path "$parent_dir/example/benchmark" -path "$parent_dir/example/sqlite_milvus_mock" \) -prune -o \( -type f \( -name 'data_map*.txt' -or -name 'faiss.index' -or -name 'sqlite.db' \) -delete \)