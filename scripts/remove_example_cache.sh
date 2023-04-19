#!/bin/bash

parent_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
find "$parent_dir/examples" \( -path "$parent_dir/examples/benchmark" -path "$parent_dir/examples/sqlite_milvus_mock" \) -prune -o \( -type f \( -name 'data_map*.txt' -or -name 'faiss.index' -or -name '*.db' \) -delete \)