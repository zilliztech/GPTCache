from gptcache.processor.post import random_one, first, nop, temperature_softmax


def test_random_one():
    message = random_one(["foo", "foo2"])
    assert message


def test_first():
    message = first(["foo", "foo2"])
    assert message == "foo"


def test_nop():
    message = nop(["foo", "foo2"])
    assert "foo" in message
    assert "foo2" in message


def test_temperature_softmax():
    message = temperature_softmax(messages=["foo", "foo2"], scores=[0.0, 1.0], temperature=0.5)
    assert message in ["foo", "foo2"]

    message = temperature_softmax(messages=["foo", "foo2"], scores=[0.9, 0.1], temperature=0.0)
    assert message == "foo"

    message = temperature_softmax(messages=["foo", "foo2"], scores=[0.1, 0.9], temperature=0.0)
    assert message == "foo2"


if __name__ == "__main__":
    test_first()
    test_nop()
    test_random_one()
    test_temperature_softmax()