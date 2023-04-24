from gptcache.utils import import_huggingface, import_torch, import_torchvision
from gptcache.embedding.base import BaseEmbedding

import_torch()
import_huggingface()
import_torchvision()

import torch  # pylint: disable=C0413
from transformers import AutoImageProcessor  # pylint: disable=C0413
from transformers import ViTModel  # pylint: disable=C0413


class ViT(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models from Huggingface transformers.

    :param model: model name, defaults to 'google/vit-base-patch16-384'.
    :type model: str

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

    def __init__(self, model: str = "google/vit-base-patch16-384"):

        self.model_name = model
        model = ViTModel.from_pretrained(model)
        self.model = model.eval()
        config = self.model.config
        self.__dimension = config.hidden_size

    def to_embeddings(self, data, **__):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        inputs = self.preprocess(data)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states[:, 0, :]
        features = features.squeeze()
        return features.detach().numpy()

    def preprocess(self, data):
        image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        inputs = image_processor(data, return_tensors="pt")
        return inputs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension

