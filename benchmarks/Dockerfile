# Reference environment for the eviction-policy benchmarks: benchmarks/results/
# is produced by running this image in `full` mode, and is not committed.
# Usage: docker compose run --rm bench [smoke|quick|full]; `--help` lists modes.

# Arch is pinned: it changes BLAS reduction order, hence embeddings, hence a
# few near-threshold similarity decisions.
FROM --platform=linux/amd64 python:3.10.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    PYTHONHASHSEED=0

# faiss and torch need a C++ runtime; git is needed by pip install -e .
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing a benchmark script does not invalidate the
# ~2 GB torch layer. setup.py reads __version__ from gptcache/__init__.py.
COPY requirements.txt setup.py README.md ./
COPY benchmarks/requirements.txt benchmarks/requirements.txt
COPY gptcache/__init__.py gptcache/__init__.py
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r benchmarks/requirements.txt

COPY . .
RUN pip install -e . \
    && chmod +x benchmarks/docker/run_pipeline.sh

VOLUME ["/app/benchmarks/data", "/root/.cache/huggingface"]

ENTRYPOINT ["benchmarks/docker/run_pipeline.sh"]
CMD ["smoke"]
