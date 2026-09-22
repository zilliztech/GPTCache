# Reproducible benchmark environment for the Cost-Aware W-TinyLFU project.
# Build:  docker build -t gptcache-bench .
# Run:    docker run --rm -v "$PWD/out:/app/out" gptcache-bench <benchmark args>
# See BENCHMARKING.md for the exact commands that regenerate each paper result.

FROM python:3.11-slim

# libgomp1 is the one system lib the faiss-cpu wheel links against at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code edits.
# - requirements.txt            : gptcache core (numpy, cachetools, requests)
# - examples/benchmark/*.txt    : tiktoken, datasets, sentence-transformers (pulls torch)
# - faiss-cpu / onnxruntime      : default vector index + default ONNX encoder (storage cell A)
COPY requirements.txt ./requirements.txt
COPY examples/benchmark/requirements.txt ./bench-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r bench-requirements.txt \
 && pip install --no-cache-dir faiss-cpu onnxruntime psutil

COPY . /app
RUN pip install --no-cache-dir -e .

# HuggingFace dataset/model cache. Mount a volume here to avoid re-downloading:
#   docker run -v "$PWD/.hf_cache:/app/.hf_cache" ...
ENV HF_HOME=/app/.hf_cache

# ultrachat is public and needs no token. LMSYS-Chat-1M and WildChat-1M are
# gated: pass a token with  docker run -e HF_TOKEN=hf_xxxxx ...
# (the datasets library reads HF_TOKEN automatically).

# Default: print the eviction benchmark's help so a bare `docker run` is self-documenting.
ENTRYPOINT ["python"]
CMD ["examples/benchmark/benchmark_lmsys.py", "--help"]
