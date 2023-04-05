from gptcache.utils import import_huggingface, import_torch
import_huggingface()
import_torch()

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Huggingface:
    """Generate sentence embedding for given text using pretrained models from Huggingface transformers.

    :param model: model name, defaults to "sentence-transformers/all-MiniLM-L6-v2".
    :type model: str

    Example:
        .. code-block:: python
        
            from gptcache.embedding import Huggingface
            
            test_sentence = "Hello, world." 
            encoder = Huggingface(model="sentence-transformers/all-MiniLM-L6-v2")
            embed = encoder.to_embeddings(test_sentence)
    """
    def __init__(self, model: str="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = '[PAD]'
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **kwargs):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        outs = self.model(**inputs).last_hidden_state
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype('float32')
    
    def post_proc(self, token_embeddings, inputs):
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embs = torch.sum(
            token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension