from gptcache.processor.post import random_one, first


def test_random_one():
    message = random_one(["foo", "foo2"])
    assert message


def test_first():
    message = first(["foo", "foo2"])
    assert message == "foo"
