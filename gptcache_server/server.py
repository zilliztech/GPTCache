import argparse
from typing import Optional

from gptcache import cache
from gptcache.adapter.api import (
    get,
    put,
    init_similar_cache,
    init_similar_cache_from_config,
)
from gptcache.utils import import_fastapi, import_pydantic

import_fastapi()
import_pydantic()

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


app = FastAPI()


class CacheData(BaseModel):
    prompt: str
    answer: Optional[str] = ""


@app.get("/")
async def hello():
    return "hello gptcache server"


@app.post("/put")
async def put_cache(cache_data: CacheData) -> str:
    put(cache_data.prompt, cache_data.answer)
    return "successfully update the cache"


@app.post("/get")
async def get_cache(cache_data: CacheData) -> CacheData:
    result = get(cache_data.prompt)
    return CacheData(prompt=cache_data.prompt, answer=result)


@app.post("/flush")
async def get_cache() -> str:
    cache.flush()
    return "successfully flush the cache"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--host", default="localhost", help="the hostname to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="the port to listen on"
    )
    parser.add_argument(
        "-d", "--cache-dir", default="gptcache_data", help="the cache data dir"
    )
    parser.add_argument(
        "-f", "--cache-config-file", default=None, help="the cache config file"
    )

    args = parser.parse_args()

    if args.cache_config_file:
        init_similar_cache_from_config(config_dir=args.cache_config_file)
    else:
        init_similar_cache(args.cache_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
