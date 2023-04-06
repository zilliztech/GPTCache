from gptcache.processor.post import random_one, first, nop


def test_random_one():
    message = random_one(['foo', 'foo2'])
    assert message


def test_first():
    message = first(['foo', 'foo2'])
    assert message == 'foo'


def test_nop():
    message = nop('foo')
    assert message == 'foo'
