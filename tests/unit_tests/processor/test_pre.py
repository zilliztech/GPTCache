from gptcache.processor.pre import (
    last_content,
    all_content,
    nop,
    last_content_without_prompt,
    get_prompt, get_openai_moderation_input,
    concat_all_queries,
    get_file_bytes,
    get_input_str,
    get_image_question,
)

import io
import os
import tempfile

from gptcache.config import Config

def test_last_content():
    content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})

    assert content == "foo2"


def test_last_content_without_prompt():
    content = last_content_without_prompt(
        {"messages": [{"content": "foo1"}, {"content": "foo2"}]}
    )
    assert content == "foo2"

    content = last_content_without_prompt(
        {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=None
    )
    assert content == "foo2"

    content = last_content_without_prompt(
        {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=["foo"]
    )
    assert content == "2"


def test_all_content():
    content = all_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})

    assert content == "foo1\nfoo2"


def test_nop():
    content = nop({"str": "hello"})
    assert content == {"str": "hello"}


def test_get_prompt():
    content = get_prompt({"prompt": "foo"})
    assert content == "foo"


def test_get_openai_moderation_input():
    content = get_openai_moderation_input({"input": ["hello", "world"]})
    assert content == "['hello', 'world']"


def test_get_messages_last_content():
    content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
    assert content == "foo2"

def test_concat_all_queries():
    config = Config()
    config.context_len = 2
    content = concat_all_queries({"messages":[{"role": "system",   "content": "foo1"}, 
                                        {"role": "user",     "content": "foo2"}, 
                                        {"role": "assistant","content": "foo3"}, 
                                        {"role": "user",     "content": "foo4"}, 
                                        {"role": "assistant","content": "foo5"},
                                        {"role": "user",     "content": "foo6"}]}, **{'cache_config':config})
    assert content == 'USER: foo4\nUSER: foo6'


if __name__  == '__main__':
    test_concat_all_queries()


# ---------- AC-2 fix: peek() → sha256(read()) ----------

SHARED_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 8188  # 8192 bytes


def _make_stream(tail: bytes) -> io.BufferedReader:
    return io.BufferedReader(io.BytesIO(SHARED_HEADER + tail))


def test_get_file_bytes_no_collision():
    """Two files sharing the same 8KB header must produce different cache keys."""
    key_a = get_file_bytes({"file": _make_stream(b"\xAA" * 4096)})
    key_b = get_file_bytes({"file": _make_stream(b"\xBB" * 4096)})
    assert key_a != key_b


def test_get_file_bytes_same_content():
    """Identical files must still produce the same cache key."""
    key_a = get_file_bytes({"file": _make_stream(b"\xAA" * 4096)})
    key_b = get_file_bytes({"file": _make_stream(b"\xAA" * 4096)})
    assert key_a == key_b


def test_get_file_bytes_resets_pointer():
    """File pointer must be at 0 after get_file_bytes so LLM can read the full file."""
    stream = _make_stream(b"\xAA" * 4096)
    get_file_bytes({"file": stream})
    assert stream.tell() == 0


def test_get_input_str_no_collision():
    question = "What is this?"
    key_a = get_input_str({"input": {"image": _make_stream(b"\xAA" * 4096), "question": question}})
    key_b = get_input_str({"input": {"image": _make_stream(b"\xBB" * 4096), "question": question}})
    assert key_a != key_b


def test_get_input_str_same_content():
    question = "What is this?"
    key_a = get_input_str({"input": {"image": _make_stream(b"\xAA" * 4096), "question": question}})
    key_b = get_input_str({"input": {"image": _make_stream(b"\xAA" * 4096), "question": question}})
    assert key_a == key_b


def test_get_input_str_different_question():
    stream_data = b"\xAA" * 4096
    key_a = get_input_str({"input": {"image": _make_stream(stream_data), "question": "Q1"}})
    key_b = get_input_str({"input": {"image": _make_stream(stream_data), "question": "Q2"}})
    assert key_a != key_b


def test_get_input_str_resets_pointer():
    stream = _make_stream(b"\xAA" * 4096)
    get_input_str({"input": {"image": stream, "question": "test"}})
    assert stream.tell() == 0


def test_get_image_question_no_collision():
    question = "What is this?"
    key_a = get_image_question({"image": _make_stream(b"\xAA" * 4096), "question": question})
    key_b = get_image_question({"image": _make_stream(b"\xBB" * 4096), "question": question})
    assert key_a != key_b


def test_get_image_question_with_filepath():
    """Test get_image_question when image is a file path string."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, SHARED_HEADER + b"\xCC" * 4096)
        os.close(fd)
        key = get_image_question({"image": path, "question": "test"})
        assert len(key) > 64  # sha256 hex (64 chars) + question
    finally:
        os.unlink(path)
