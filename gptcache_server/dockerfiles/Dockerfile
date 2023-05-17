FROM python:3.8-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir gptcache

WORKDIR /workspace

CMD ["gptcache_server", "-s", "0.0.0.0"]
