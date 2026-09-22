import os

import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_huggingface()
import_onnxruntime()
import_huggingface_hub()

from transformers import AutoTokenizer, AutoConfig  # pylint: disable=C0413
from huggingface_hub import hf_hub_download  # pylint: disable=C0413
import onnxruntime  # pylint: disable=C0413


def _is_dynamic_batch(session):
    """Return True if the ONNX session accepts a dynamic batch dimension."""
    for inp in session.get_inputs():
        shape = inp.shape
        if shape and shape[0] not in (1, None) and not isinstance(shape[0], str):
            return False  # static batch > 1 or unexpected
        if shape and (shape[0] is None or isinstance(shape[0], str)):
            return True   # symbolic / dynamic
    # If batch dim is explicitly 1, it's static
    for inp in session.get_inputs():
        if inp.shape and inp.shape[0] == 1:
            return False
    return True  # assume dynamic if unclear


class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.

    Supports both the original static-batch-1 model from HuggingFace Hub and
    a locally re-exported dynamic-batch model (see scripts/export_onnx_dynamic.py).

    Set the environment variable ``GPTCACHE_ONNX_MODEL_DIR`` to the directory
    containing a re-exported ``model.onnx`` with dynamic batch axes to enable
    true batched inference (~30-50x faster ingest).

    Example:
        .. code-block:: python

            from gptcache.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model='GPTCache/paraphrase-albert-onnx')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="GPTCache/paraphrase-albert-onnx"):
        tokenizer_name = "GPTCache/paraphrase-albert-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model

        # Allow a locally re-exported dynamic-batch model to be used instead
        # of the static-batch-1 model from HuggingFace Hub.
        local_model_dir = os.environ.get("GPTCACHE_ONNX_MODEL_DIR", "")
        if local_model_dir:
            onnx_model_path = os.path.join(local_model_dir, "model.onnx")
            if not os.path.isfile(onnx_model_path):
                raise FileNotFoundError(
                    f"GPTCACHE_ONNX_MODEL_DIR is set to '{local_model_dir}' "
                    f"but model.onnx was not found there. "
                    f"Run scripts/export_onnx_dynamic.py first."
                )
        else:
            onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self._dynamic_batch = _is_dynamic_batch(self.ort_session)

        config = AutoConfig.from_pretrained(
            "GPTCache/paraphrase-albert-small-v2"
        )
        self.__dimension = config.hidden_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts):
        """Tokenize a list of strings. Returns numpy arrays of shape (N, 512)."""
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_token_type_ids=True,
        )
        input_ids = np.array(encoded["input_ids"]).astype("int64")
        attention_mask = np.array(encoded["attention_mask"]).astype("int64")
        token_type_ids = np.array(
            encoded.get("token_type_ids", [[0] * 512] * len(texts))
        ).astype("int64")
        return input_ids, attention_mask, token_type_ids

    def _run_onnx(self, input_ids, attention_mask, token_type_ids):
        """Run the ONNX session and return mean-pooled embeddings (N, dim)."""
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        ort_feat = self.ort_session.run(None, ort_inputs)[0]  # (N, seq, dim)
        return self.post_proc(ort_feat, attention_mask)        # (N, dim)

    def _encode_batch_dynamic(self, texts):
        """Encode a list of strings in a single batched ONNX call."""
        input_ids, attention_mask, token_type_ids = self._tokenize(texts)
        return self._run_onnx(input_ids, attention_mask, token_type_ids)  # (N, dim)

    def _encode_batch_static(self, texts):
        """Encode a list of strings one-by-one (static batch-size-1 model)."""
        results = []
        for text in texts:
            ids, mask, ttids = self._tokenize([text])
            results.append(self._run_onnx(ids, mask, ttids))  # (1, dim)
        return np.vstack(results)  # (N, dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string or list of strings.
        :type data: str or list[str]

        :return: a text embedding in shape of (dim,) for a single string,
                 or (N, dim) for a batch.
        """
        is_single = isinstance(data, str)
        texts = [data] if is_single else list(data)

        if self._dynamic_batch:
            emb = self._encode_batch_dynamic(texts)  # (N, dim)
        else:
            emb = self._encode_batch_static(texts)   # (N, dim)

        return emb.flatten() if is_single else emb

    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
