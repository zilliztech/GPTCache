# Security Vulnerability Report: Cache Poisoning via `peek()` Collision

| Field | Value |
|---|---|
| **Report ID** | AC-2 |
| **Title** | Image/File Cache Key Collision Leading to Cache Poisoning |
| **Severity** | High (CVSS 3.1 Base Score: 7.5) |
| **CVSS Vector** | AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N |
| **Affected Component** | `gptcache/processor/pre.py` |
| **Affected Versions** | All versions (as of commit `c59fb3a`) |
| **Date** | 2026-03-25 |

---

## 1. Summary

GPTCache uses Python's `BufferedReader.peek()` to generate cache keys for image and file inputs. `peek()` only returns the internal buffer content (typically the first **8192 bytes**), not the full file. An attacker can construct two files that share the same first 8192 bytes but contain entirely different content, causing the cache to treat them as identical. This enables **cache poisoning** and **information disclosure**.

---

## 2. Affected Functions

Three functions in [`gptcache/processor/pre.py`](../../gptcache/processor/pre.py) are affected:

### 2.1 `get_input_str()` (Line 245-246)

Used by: Replicate adapter ([`gptcache/adapter/replicate.py`](../../gptcache/adapter/replicate.py))

```python
def get_input_str(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]  # <-- vulnerability
```

### 2.2 `get_file_bytes()` (Line 229)

Used by: OpenAI audio transcription adapter ([`gptcache/adapter/openai.py:248`](../../gptcache/adapter/openai.py))

```python
def get_file_bytes(data: Dict[str, Any], **_: Dict[str, Any]) -> bytes:
    return data.get("file").peek()  # <-- vulnerability
```

### 2.3 `get_image_question()` (Line 280-282)

Used by: MiniGPT4 adapter ([`gptcache/adapter/minigpt4.py`](../../gptcache/adapter/minigpt4.py))

```python
def get_image_question(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    img = data.get("image")
    data_img = str(open(img, "rb").peek()) if isinstance(img, str) else str(img)
    return data_img + data.get("question")  # <-- vulnerability
```

---

## 3. Root Cause

Python's `BufferedReader.peek()` is designed to "peek" at the internal read buffer **without advancing the file pointer**. Its behavior:

- Returns **at most** the contents of the internal buffer (default size: **8192 bytes**)
- Does **NOT** read the full file, regardless of file size
- For a 1 MB file, `peek()` returns only 0.78% of the content

The vulnerable functions use `peek()` output as the cache key (or as input to the embedding function that generates the cache key). Since the cache key is derived from an incomplete representation of the file, files with identical prefixes but different content map to the same cache entry.

---

## 4. Attack Scenario

### Prerequisites

- The attacker can send requests to a GPTCache-enabled endpoint that processes image or file inputs
- The cache is shared (multi-user, or attacker can access the same cache instance)

### Attack Steps

```
Step 1 ─ Prime the cache
    Attacker sends: img_A (legitimate image) + question_Q
    → Cache MISS → LLM processes full img_A → answer_A cached
    → Cache key = str(peek(img_A)) + question_Q

Step 2 ─ Exploit the collision
    Attacker constructs img_B:
      - First 8192 bytes identical to img_A (copy JPEG/PNG header)
      - Remaining bytes contain completely different (malicious) content
    Attacker sends: img_B + question_Q
    → pre_embedding_func: str(peek(img_B)) + question_Q
    → peek(img_B) == peek(img_A)  [first 8192 bytes match]
    → Cache key identical → Cache HIT
    → Returns answer_A (the answer for img_A, NOT img_B)
    → LLM is never called for img_B
```

### Data Flow Diagram

```
              User Request
                  │
                  ▼
    ┌─────────────────────────┐
    │    adapt() in adapter.py │
    │                         │
    │  ┌───────────┐          │
    │  │ pre_embed  │──peek()──│──→ Only 8192 bytes → Cache Key
    │  │ _func()   │          │         │
    │  └───────────┘          │         ▼
    │         │               │    Cache Lookup
    │         │               │    (HIT if prefix matches)
    │         ▼               │         │
    │  ┌──────────────┐       │    HIT? ──Yes──→ Return cached answer
    │  │ llm_handler() │       │         │         (WRONG answer!)
    │  │ seek(0)+read()│       │    No───→ Call LLM with full file
    │  │ (full file)   │       │
    │  └──────────────┘       │
    └─────────────────────────┘
```

### Impact

| Impact Type | Description |
|---|---|
| **Cache Poisoning** | Queries for img_B return img_A's answer. All subsequent requests with the same peek prefix are affected. |
| **Information Disclosure** | Attacker can probe cached answers for other users' images by constructing files with matching prefixes. |
| **Persistent** | Poisoned entries remain until cache eviction or manual cleanup. |
| **Cross-User** | In shared cache deployments, all users are affected. |

---

## 5. Proof of Concept

Two PoC scripts are provided in this directory:

### 5.1 Cache Key Collision Test

**File:** [`poc_ac2_peek_collision.py`](../poc_ac2_peek_collision.py)

Demonstrates that `peek()` returns identical results for files with the same 8192-byte prefix but different content.

**Result:**

```
  get_input_str (small ~8KB)  : VULNERABLE
  get_input_str (large 1MB)   : VULNERABLE
  get_file_bytes (large 1MB)  : VULNERABLE
  get_image_question (large)  : VULNERABLE

  4/4 vectors confirmed exploitable.
```

### 5.2 End-to-End Cache Poisoning Test

**File:** [`poc_ac2_e2e_poisoning.py`](../poc_ac2_e2e_poisoning.py)

Demonstrates the full attack chain using GPTCache's `Cache`, `SSDataManager` (SQLite + FAISS), and `ExactMatchEvaluation`.

**Result:**

```
  img_A content hash: 08b95537eea9fa4f...
  img_B content hash: 6e8a133a461377eb...
  Images identical  : NO (completely different after byte 8192)

  Cache key(img_A)  : e5b58d2951bfcad4...
  Cache key(img_B)  : e5b58d2951bfcad4...
  Keys identical    : YES

  Similarity score  : 1.0
  img_B returned img_A's answer: YES → CACHE POISONING CONFIRMED
```

### 5.3 Reproduction Steps

```bash
# From repository root
pip install cachetools requests sqlalchemy faiss-cpu numpy

# Test 1: Cache key collision
python tests/poc_ac2_peek_collision.py

# Test 2: End-to-end cache poisoning
python tests/poc_ac2_e2e_poisoning.py
```

---

## 6. Additional Observations

### 6.1 No Input Validation

There is **no file size limit, format validation, or content sanitization** anywhere in the input processing chain:

| Layer | File | Validation |
|---|---|---|
| Pre-processing | `gptcache/processor/pre.py` | None |
| Adapter | `gptcache/adapter/adapter.py` | None |
| Config | `gptcache/config.py` | None (only `similarity_threshold`, `max_size` for cache entry count) |

### 6.2 Asymmetric Read Behavior

The cache key path and the LLM call path read the file differently:

| Path | Method | Bytes Read |
|---|---|---|
| Cache key generation | `peek()` | 8192 (fixed, buffer size) |
| LLM invocation (Replicate SDK) | `seek(0)` + `read()` | Full file |

This asymmetry is the fundamental design flaw: the cache key does not represent the data that the LLM actually processes.

### 6.3 Ease of Exploitation

Constructing colliding files is trivial for common image formats:

| Format | Fixed Header Size | Collision Method |
|---|---|---|
| JPEG | `FF D8 FF` + APP markers + EXIF (typically 2-8 KB) | Copy EXIF metadata block |
| PNG | 8-byte signature + IHDR (25 bytes) + partial IDAT | Same dimensions + color mode |
| WAV | 44-byte header + initial samples | Same sample rate + channels |
| MP3 | ID3 tag + initial frames | Copy ID3 tag |

---

## 7. Suggested Fix

### Option A: Hash Full Content (Recommended)

Replace `peek()` with `read()` + cryptographic hash, then reset the file pointer:

```python
import hashlib

def get_input_str(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    image = input_data["image"]
    content = image.read()
    image.seek(0)  # reset for downstream LLM consumption
    image_hash = hashlib.sha256(content).hexdigest()
    return image_hash + input_data["question"]


def get_file_bytes(data: Dict[str, Any], **_: Dict[str, Any]) -> bytes:
    f = data.get("file")
    content = f.read()
    f.seek(0)
    return hashlib.sha256(content).hexdigest()


def get_image_question(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    img = data.get("image")
    if isinstance(img, str):
        with open(img, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()
    else:
        content = img.read()
        img.seek(0)
        img_hash = hashlib.sha256(content).hexdigest()
    return img_hash + data.get("question")
```

### Option B: Streaming Hash (For Large Files)

```python
def _hash_file(f, chunk_size=65536) -> str:
    h = hashlib.sha256()
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)
    f.seek(0)
    return h.hexdigest()
```

### Considerations

| Approach | Pros | Cons |
|---|---|---|
| Option A | Simple, direct | Loads full file into memory |
| Option B | Memory-efficient | Slightly more complex |

Both options fully resolve the collision vulnerability by ensuring the **entire** file content participates in cache key generation.

---

## 8. References

- [Python docs: `BufferedReader.peek()`](https://docs.python.org/3/library/io.html#io.BufferedReader.peek)
  > *"Return buffered data without advancing the position. At most a single read on the raw stream is done to satisfy the call. The number of bytes returned may be less or more than requested."*
- [OWASP: Cache Poisoning](https://owasp.org/www-community/attacks/Cache_Poisoning)
- GPTCache source: https://github.com/zilliztech/GPTCache
