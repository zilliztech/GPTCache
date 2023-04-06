from gptcache.utils import import_onnxruntime, import_huggingface_hub, import_huggingface
from .similarity_evaluation import SimilarityEvaluation

import_onnxruntime()
import_huggingface_hub()
import_huggingface()
import numpy as np
import os
from transformers import AutoTokenizer
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import List
import onnxruntime

def pad_sequence(input_ids_list, padding_value=0):
    max_len = max(len(sequence) for sequence in input_ids_list)
    padded_sequences = np.full((len(input_ids_list), max_len), padding_value)
    for i, sequence in enumerate(input_ids_list):
        padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences

class Onnx(SimilarityEvaluation):
    def __init__(self, model = 'GPTCache/albert-duplicate-onnx'):
        tokenizer_name = 'albert-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename = 'model.onnx')
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # WARNING: the model cannot evaluate text with more than 512 tokens
    def evaluation(self, src_dict, cache_dict, **kwargs):
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            if src_question == cache_question:
                return 1
            return self.inference(src_question, [cache_question])
        except Exception:
            return 0

    def range(self):
        return 0.0, 1.0

    def inference(self, reference: str, candidates: List[str]) -> np.ndarray:
 
         n_candidates= len(candidates)
         inference_texts = [{'text_a': reference, 'text_b': candidate } for candidate in  candidates ]
         batch_encoding_list = [self.tokenizer.encode_plus(text['text_a'], text['text_b'], padding='longest') for text in inference_texts]
 
         input_ids_list = [np.array(encode.input_ids) for encode in batch_encoding_list] 
         attention_mask_list = [np.array(encode.attention_mask) for encode in batch_encoding_list] 
         token_type_ids_list = [np.array(encode.token_type_ids) for encode in batch_encoding_list]
 
         padded_input_ids = pad_sequence(input_ids_list, padding_value=self.tokenizer.pad_token_id)
         padded_attention_mask = pad_sequence(attention_mask_list, padding_value=self.tokenizer.pad_token_id)
         padded_token_type_ids = pad_sequence(token_type_ids_list, padding_value=self.tokenizer.pad_token_id)
 
         ort_inputs = {'input_ids': padded_input_ids.reshape(n_candidates,-1), 
                       'attention_mask': padded_attention_mask.reshape(n_candidates,-1),
                       'token_type_ids': padded_token_type_ids.reshape(n_candidates,-1)}
         ort_outputs = self.ort_session.run(None, ort_inputs)
         scores = ort_outputs[0][:,1]
         return scores
