# Reproducible environment for the eviction-policy benchmarks.
#
#   docker build -f benchmarks/Dockerfile -t gptcache-arc-bench .
#   docker run --rm -it \
#       -v "$PWD/benchmarks/results:/app/benchmarks/results" \
#       -v gptcache-hf-cache:/root/.cache/huggingface \
#       gptcache-arc-bench
#
# The default command runs the whole pipeline: tests, data prep, gate, sweep,
# figures. Mount the HF cache volume as above so the ~700 MB of datasets and
# the embedding model are downloaded once rather than on every run.
#
# CPU only, so the numbers are portable. On an M-series Mac, running natively
# (see benchmarks/README.md) uses the MPS backend and prepares data several
# times faster; the hit rates are identical either way, the wall-clock and
# latency figures are not.

FROM python:3.10.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=1

# faiss and torch need a C++ runtime; git is needed by pip install -e .
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so a source edit does not invalidate the install layer
COPY requirements.txt setup.py README.md ./
COPY benchmarks/requirements.txt benchmarks/requirements.txt
COPY gptcache/__init__.py gptcache/__init__.py
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r benchmarks/requirements.txt

COPY . .
RUN pip install -e .

CMD ["bash", "-lc", "\
    python -m pytest tests/unit_tests/eviction/ tests/unit_tests/manager/test_eviction.py -q && \
    python benchmarks/prepare_data.py --device cpu && \
    python benchmarks/gate_screening.py && \
    python benchmarks/run_bench.py --jobs 4 && \
    python benchmarks/analyze.py"]
