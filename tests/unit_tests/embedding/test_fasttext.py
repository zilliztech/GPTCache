from unittest.mock import patch

from gptcache.embedding import FastText

from gptcache.utils import import_fasttext
import_fasttext()

import fasttext


def test_embedding():
    with patch("fasttext.util.download_model") as download_model_mock:
        download_model_mock.return_value = "fastttext.bin"
        with patch("fasttext.load_model") as load_model_mock:
            load_model_mock.return_value = fasttext.FastText._FastText()
            with patch("fasttext.util.reduce_model") as reduce_model_mock:
                reduce_model_mock.return_value = None
                with patch("fasttext.FastText._FastText.get_dimension") as dimension_mock:
                    dimension_mock.return_value = 128
                    with patch("fasttext.FastText._FastText.get_sentence_vector") as vector_mock:
                        vector_mock.return_value = [0] * 128

                        ft = FastText(dim=128)
                        assert len(ft.to_embeddings("foo")) == 128
                        assert ft.dimension == 128
