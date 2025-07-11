from gptcache.processor.post import random_one, first, nop, temperature_softmax
from unittest.mock import Mock


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


def test_llm_verifier():
    # mock client that always returns 'yes'
    mock_client = Mock()
    mock_resp = Mock()
    mock_choice = Mock()
    mock_choice.message.content = 'yes'
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    from gptcache.processor.post import LlmVerifier
    verifier = LlmVerifier(client=mock_client, system_prompt="test prompt", model="fake-model")
    messages = ["foo", "bar"]
    scores = [0.1, 0.9]
    result = verifier(messages, scores=scores, original_question="test question")
    assert result == "bar"

    # mock client that returns 'no'
    mock_choice.message.content = 'no'
    result = verifier(messages, scores=scores, original_question="test question")
    assert result is None




if __name__ == "__main__":
    test_first()
    test_nop()
    test_random_one()
    test_temperature_softmax()
    test_llm_verifier()
