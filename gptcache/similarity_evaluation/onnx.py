from typing import Dict, List, Tuple, Any

import numpy as np

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_onnxruntime()
import_huggingface_hub()
import_huggingface()

from transformers import AutoTokenizer  # pylint: disable=C0413
from huggingface_hub import hf_hub_download  # pylint: disable=C0413
import onnxruntime  # pylint: disable=C0413


def pad_sequence(input_ids_list: List[np.ndarray], padding_value: int = 0):
    max_len = max(len(sequence) for sequence in input_ids_list)
    padded_sequences = np.full((len(input_ids_list), max_len), padding_value)
    for i, sequence in enumerate(input_ids_list):
        padded_sequences[i, : len(sequence)] = sequence
    return padded_sequences


class OnnxModelEvaluation(SimilarityEvaluation):
    """Using ONNX model to evaluate sentences pair similarity.

    This evaluator use the ONNX model to evaluate the similarity of two sentences.

    :param model: model name of OnnxModelEvaluation. Default is 'GPTCache/albert-duplicate-onnx'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import OnnxModelEvaluation

            evaluation = OnnxModelEvaluation()
            score = evaluation.evaluation(
                {
                    'question': 'What is the color of sky?'
                },
                {
                    'question': 'hello'
                }
            )
    """

    def __init__(self, model: str = "GPTCache/albert-duplicate-onnx"):
        tokenizer_name = "albert-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # WARNING: the model cannot evaluate text with more than 512 tokens
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            if src_question.lower() == cache_question.lower():
                return 1
            return self.inference(src_question, [cache_question])
        except Exception:  # pylint: disable=W0703
            return 0

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 1.0

    def inference(self, reference: str, candidates: List[str]) -> np.ndarray:
        """Inference the ONNX model.

        :param reference: reference sentence.
        :type reference: str
        :param candidates: candidate sentences.
        :type candidates: List[str]

        :return: probability score indcates how much is reference similar to candidates.
        """
        n_candidates = len(candidates)
        inference_texts = [
            {"text_a": reference, "text_b": candidate} for candidate in candidates
        ]
        batch_encoding_list = [
            self.tokenizer.encode_plus(
                text["text_a"], text["text_b"], padding="longest"
            )
            for text in inference_texts
        ]

        input_ids_list = [np.array(encode.input_ids) for encode in batch_encoding_list]
        attention_mask_list = [
            np.array(encode.attention_mask) for encode in batch_encoding_list
        ]
        token_type_ids_list = [
            np.array(encode.token_type_ids) for encode in batch_encoding_list
        ]

        padded_input_ids = pad_sequence(
            input_ids_list, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_mask = pad_sequence(
            attention_mask_list, padding_value=self.tokenizer.pad_token_id
        )
        padded_token_type_ids = pad_sequence(
            token_type_ids_list, padding_value=self.tokenizer.pad_token_id
        )

        ort_inputs = {
            "input_ids": padded_input_ids.reshape(n_candidates, -1),
            "attention_mask": padded_attention_mask.reshape(n_candidates, -1),
            "token_type_ids": padded_token_type_ids.reshape(n_candidates, -1),
        }
        ort_outputs = self.ort_session.run(None, ort_inputs)
        scores = ort_outputs[0][:, 1]
        return float(scores[0])
