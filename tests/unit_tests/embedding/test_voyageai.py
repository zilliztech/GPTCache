import os
import types
import pytest
import mock
from gptcache.utils import import_voyageai
from gptcache.embedding import VoyageAI

import_voyageai()



@mock.patch.dict(os.environ, {"VOYAGE_API_KEY": "API_KEY", "VOYAGE_API_KEY_PATH": "API_KEY_FILE_PATH_ENV"})
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="API_KEY")
@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_without_api_key(mock_created, mock_file):
    dimension = 1024
    vo = VoyageAI()

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension

    mock_file.assert_called_once_with("API_KEY_FILE_PATH_ENV", "rt")
    mock_created.assert_called_once_with(texts=["foo"], model="voyage-3", input_type=None, truncation=True)


@mock.patch.dict(os.environ, {"VOYAGE_API_KEY": "API_KEY", "VOYAGE_API_KEY_PATH": "API_KEY_FILE_PATH_ENV"})
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="API_KEY")
@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_api_key_path(mock_create, mock_file):
    dimension = 1024
    vo = VoyageAI(api_key_path="API_KEY_FILE_PATH")

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension

    mock_file.assert_called_once_with("API_KEY_FILE_PATH", "rt")
    mock_create.assert_called_once_with(texts=["foo"], model="voyage-3", input_type=None, truncation=True)


@mock.patch.dict(os.environ, {"VOYAGE_API_KEY": "API_KEY"})
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="API_KEY")
@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_api_key_in_envrion(mock_create, mock_file):
    dimension = 1024
    vo = VoyageAI()

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_file.assert_not_called()
    mock_create.assert_called_once_with(texts=["foo"], model="voyage-3", input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_api_key(mock_create):
    dimension = 1024
    vo = VoyageAI(api_key="API_KEY")

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model="voyage-3", input_type=None, truncation=True)


@mock.patch.dict(os.environ)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="API_KEY")
def test_voageai_without_api_key_or_api_key_file_path(mock_file):
    with pytest.raises(Exception):
        VoyageAI()
    mock_file.assert_not_called()


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 512]))
def test_voageai_with_model_voyage_3_lite(mock_create):
    dimension = 512
    model = "voyage-3-lite"
    vo = VoyageAI(api_key="API_KEY", model=model)

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_model_voyage_finance_2(mock_create):
    dimension = 1024
    model = "voyage-finance-2"
    vo = VoyageAI(api_key="API_KEY", model=model)

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_model_voyage_multilingual_2(mock_create):
    dimension = 1024
    model = "voyage-multilingual-2"
    vo = VoyageAI(api_key="API_KEY", model=model)

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1024]))
def test_voageai_with_model_voyage_law_2(mock_create):
    dimension = 1024
    model = "voyage-law-2"
    vo = VoyageAI(api_key="API_KEY", model=model)

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1536]))
def test_voageai_with_model_voyage_code_2(mock_create):
    dimension = 1536
    model = "voyage-code-2"
    vo = VoyageAI(api_key="API_KEY", model=model)

    assert vo.dimension == dimension
    assert len(vo.to_embeddings("foo")) == dimension
    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=None, truncation=True)


@mock.patch("voyageai.Client.embed", return_value=types.SimpleNamespace(embeddings=[[0] * 1536]))            
def test_voageai_with_general_parameters(mock_create):
    dimension = 1536
    model = "voyage-code-2"
    api_key = "API_KEY"
    input_type = "query"
    truncation = False

    mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])

    vo = VoyageAI(model=model, api_key=api_key, input_type=input_type, truncation=truncation)
    assert vo.dimension == dimension
    assert len(vo.to_embeddings(["foo"])) == dimension

    mock_create.assert_called_once_with(texts=["foo"], model=model, input_type=input_type, truncation=truncation)
