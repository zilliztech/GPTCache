import argparse
import json
import os
import zipfile
from typing import Optional

from gptcache import cache, Cache
from gptcache.adapter import openai
from gptcache.adapter.api import (
    get,
    put,
    init_similar_cache,
    init_similar_cache_from_config,
)
from gptcache.processor.pre import last_content
from gptcache.utils import import_fastapi, import_pydantic, import_starlette

import_fastapi()
import_pydantic()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel


app = FastAPI()
openai_cache: Optional[Cache] = None
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


@app.api_route(
    "/v1/chat/completions",
    methods=["POST", "OPTIONS"],
)
async def chat(request: Request):
    if openai_cache is None:
        raise HTTPException(
            status_code=500,
            detail=f"the gptcache server doesn't open the openai completes proxy",
        )

    import_starlette()
    from starlette.responses import StreamingResponse, JSONResponse

    openai_params = await request.json()
    is_stream = openai_params.get("stream", False)
    headers = request.headers
    auth_header = headers.get("authorization", None)
    openai_key = auth_header.split(" ")[1] if auth_header else ""
    cache_skip = openai_params.pop("cache_skip", False)
    if cache_skip is False:
        messages = openai_params.get("messages")
        if "/cache_skip " in messages[0]["content"]:
            cache_skip = True
            content0 = openai_params.get("messages")[0]["content"]
            openai_params.get("messages")[0]["content"] = str(content0).replace("/cache_skip ", "")
        elif "/cache_skip " in messages[-1]["content"]:
            cache_skip = True
            content0 = openai_params.get("messages")[-1]["content"]
            openai_params.get("messages")[-1]["content"] = str(content0).replace("/cache_skip ", "")
        print("cache_skip:", cache_skip)
    print("messages:", openai_params.get("messages"))
    try:
        if is_stream:
            def generate():
                for stream_response in openai.ChatCompletion.create(
                    cache_obj=openai_cache,
                    cache_skip=cache_skip,
                    api_key=openai_key,
                    **openai_params,
                ):
                    if stream_response == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {json.dumps(stream_response)}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            openai_response = openai.ChatCompletion.create(
                cache_obj=openai_cache,
                cache_skip=cache_skip,
                api_key=openai_key,
                **openai_params,
            )
            return JSONResponse(content=openai_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai error: {e}")


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
    parser.add_argument(
        "-o",
        "--openai",
        type=bool,
        default=False,
        help="whether to open the openai completes proxy",
    )
    parser.add_argument(
        "-of",
        "--openai-cache-config-file",
        default=None,
        help="the cache config file of the openai completes proxy",
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

    if args.openai:
        global openai_cache
        openai_cache = Cache()
        if args.openai_cache_config_file:
            init_similar_cache_from_config(
                config_dir=args.openai_cache_config_file,
                cache_obj=openai_cache,
            )
        else:
            init_similar_cache(
                data_dir="openai_server_cache",
                pre_func=last_content,
                cache_obj=openai_cache,
            )

        import_starlette()
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
