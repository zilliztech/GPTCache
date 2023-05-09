FROM ubuntu:20.04

RUN apt-get update          && \
    apt-get install -y         \
              zip              \
              curl             \
              wget             \
              unzip            \
              python3.8        \
              python3-pip      \
              libgl1-mesa-glx

RUN pip3 install -U pip

RUN pip3 install gptcache

WORKDIR /workspace

CMD ["gptcache_server", "-s", "0.0.0.0"]
