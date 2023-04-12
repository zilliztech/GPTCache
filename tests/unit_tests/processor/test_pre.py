from gptcache.processor.pre import (
    last_content,
    all_content,
    nop,
    last_content_without_prompt,
    get_prompt,
)


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
