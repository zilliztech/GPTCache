import argparse
import os
import zipfile
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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel


app = FastAPI()

cache_dir = ""
cache_file_key = ""


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


@app.get("/cache_file")
async def get_cache_file(key: str = "") -> FileResponse:
    global cache_dir
    global cache_file_key
    if cache_dir == "":
        raise HTTPException(
            status_code=403,
            detail="the cache_dir was not specified when the service was initialized",
        )
    if cache_file_key == "":
        raise HTTPException(
            status_code=403,
            detail="the cache file can't be downloaded because the cache-file-key was not specified",
        )
    if cache_file_key != key:
        raise HTTPException(status_code=403, detail="the cache file key is wrong")
    zip_filename = cache_dir + ".zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cache_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    return FileResponse(zip_filename)


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
    parser.add_argument("-k", "--cache-file-key", default="", help="the cache file key")
    parser.add_argument(
        "-f", "--cache-config-file", default=None, help="the cache config file"
    )

    args = parser.parse_args()
    global cache_dir
    global cache_file_key

    if args.cache_config_file:
        init_conf = init_similar_cache_from_config(config_dir=args.cache_config_file)
        cache_dir = init_conf.get("storage_config", {}).get("data_dir", "")
    else:
        init_similar_cache(args.cache_dir)
        cache_dir = args.cache_dir
    cache_file_key = args.cache_file_key

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
