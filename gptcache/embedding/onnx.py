from gptcache.utils import import_onnxruntime, import_huggingface_hub, import_huggingface
import_huggingface()
import_onnxruntime()
import_huggingface_hub()
import numpy as np
import os
from transformers import AutoTokenizer, AutoConfig
from pathlib import Path
from huggingface_hub import hf_hub_download
import onnxruntime

class Onnx:
    """Generate text embedding for given text using ONNX Model.

    Example:
        .. code-block:: python
        
            from gptcache.embedding import onnx
            
            test_sentence = "Hello, world." 
            encoder = onnx(model="GPTCache/paraphrase-albert-onnx")
            embed = encoder.to_embeddings(test_sentence)
    """
    def __init__(self, api_key, model= "GPTCache/paraphrase-albert-onnx", **kwargs):
        tokenizer_name = 'sentence-transformers/paraphrase-albert-small-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename = 'model.onnx')
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        config = AutoConfig.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')
        self.__dimension = config.hidden_size

    
    def to_embeddings(self, data, **kwargs):
        encoded_text = self.tokenizer.encode_plus(data, padding='max_length')
        ort_inputs = {'input_ids': np.array(encoded_text['input_ids']).reshape(1,-1), 
               'attention_mask': np.array(encoded_text['attention_mask']).reshape(1,-1),
               'token_type_ids': np.array(encoded_text['token_type_ids']).reshape(1,-1)}

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs['attention_mask'])
        return emb.flatten()

    def post_proc(self, token_embeddings, attention_mask): 
        token_embeddings = token_embeddings
        input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(token_embeddings.shape[-1], -1).astype(float)
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(input_mask_expanded.sum(1), 1e-9)
        return sentence_embs

    @property
    def dimension(self):
        return self.__dimension

