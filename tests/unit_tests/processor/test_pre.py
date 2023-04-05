from gptcache.processor.pre import last_content, all_content, nop


def test_last_content():
    content = last_content({
        "messages": [
            {
                "content": "foo1"
            },
            {
                "content": "foo2"
            }
        ]
    })

    assert content == "foo2"


def test_all_content():
    content = all_content({
        "messages": [
            {
                "content": "foo1"
            },
            {
                "content": "foo2"
            }
        ]
    })

    assert content == "foo1\nfoo2"


def test_nop():
    content = nop("hello")
    assert content == "hello"
