from unittest.mock import patch, Mock

from gptcache.client import Client


def test_client():
    client = Client()
    with patch("httpx.AsyncClient.put") as mock_response:
        mock_response.return_value = Mock(status_code=200)
        status_code = client.put("Hi", "Hi back")
        assert status_code == 200
    
    with patch("httpx.AsyncClient.get") as mock_response:
        m = Mock()
        attrs = {"json.return_value": "Hi back"}
        m.configure_mock(**attrs)
        mock_response.return_value = m
        ans = client.get("Hi")
        assert ans == "Hi back"
