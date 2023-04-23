import argparse
import http.server
import json

from gptcache.adapter.api import get, put, init_similar_cache


class GPTCacheHandler(http.server.BaseHTTPRequestHandler):
    # curl -X GET  "http://localhost:8000?prompt=hello"
    def do_GET(self):
        params = self.path.split("?")[1]
        prompt = params.split("=")[1]

        result = get(prompt)

        response = json.dumps(result)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(response, "utf-8"))

    # curl -X PUT -d "receive a hello message" "http://localhost:8000?prompt=hello"
    def do_PUT(self):
        params = self.path.split("?")[1]
        prompt = params.split("=")[1]
        content_length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(content_length).decode("utf-8")

        put(prompt, data)

        self.send_response(200)
        self.end_headers()


def start_server(host: str, port: int):
    httpd = http.server.HTTPServer((host, port), GPTCacheHandler)
    print(f"Starting server at {host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="the hostname to listen on")
    parser.add_argument("--port", type=int, default=8000, help="the port to listen on")
    parser.add_argument("--cache-dir", default="gptcache_data", help="the cache data dir")
    args = parser.parse_args()

    init_similar_cache(args.cache_dir)
    start_server(args.host, args.port)
