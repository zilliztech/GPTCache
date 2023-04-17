import numpy as np

from gptcache.utils import import_huggingface, import_torch, import_torchvision
from gptcache.embedding.base import BaseEmbedding

import_torch()
import_huggingface()
import_torchvision()

import torch # pylint: disable=C0413
from torchvision import transforms

from transformers import AutoImageProcessor
from transformers import ViTModel


class ImageEmbedding(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models from Huggingface transformers.

    :param model: model name, defaults to 'google/vit-base-patch16-384'.
    :type model: str
    :param image_processor: the 

    Example:
        .. code-block:: python
            import io
            from PIL import Image
            from gptcache.embedding import ImageEmbedding

            def prepare_image(image_data: str = None):
                if not image_data:
                    image_data = io.BytesIO()
                    Image.new('RGB', (244, 244), color=(255, 0, 0)).save(image_data, format='JPEG')
                    image_data.seek(0)
                image = Image.open(image_data)
                return image

            image = prepare_image()
            encoder = ImageEmbeddings(model="google/vit-base-patch16-384")
            embed = encoder.to_embeddings(image)
    """

    def __init__(self, 
                    model: str = "google/vit-base-patch16-384",
                    image_processor: callable = None, # NOTE: Design CHOICE: image_preprocessor is more aligned with the transformers and the literature, but in cachgpt 'pre_proc' is mostly used. 
                    ):
        
        

        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(model)
        self.image_processor = image_processor

        vit_model = ViTModel.from_pretrained(model)
        self.vit_model = vit_model.eval()
        
        config = self.vit_model.config
        self.__dimension = config.hidden_size

    def to_embeddings(self, data, **__):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        inputs = self.image_processor(data, return_tensors="pt")

        with torch.no_grad():
            outputs = self.vit_model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states[:, 0, :]
        features = features.squeeze()

        return features.detach().numpy()


    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension   
