import numpy as np

from gptcache.utils import import_timm, import_torch
from gptcache.embedding.base import BaseEmbedding

import_torch()
import_timm()

import torch  # pylint: disable=C0413
from timm.models import create_model  # pylint: disable=C0413


class Timm(BaseEmbedding):
    """Generate image embedding for given image using pretrained models from Timm.

    :param model: model name, defaults to 'resnet34'.
    :type model: str

    Example:
        .. code-block:: python

            import requests
            from PIL import Image
            from gptcache.embedding import Timm


            url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
            image = Image.open(requests.get(url, stream=True).raw)  # Read image url as PIL.Image

            encoder = Timm(model='resnet50')
            image_tensor = encoder.preprocess(image)
            embed = encoder.to_embeddings(image_tensor)
    """

    def __init__(self, model: str = "resnet18", device: str = "default"):
        if device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name = model
        self.model = create_model(model_name=model, pretrained=True)
        self.model.eval()

        try:
            self.__dimension = self.model.embed_dim
        except Exception:  # pylint: disable=W0703
            self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given image data

        :param data: image data with batch size at index 0.
        :type data: torch.Tensor

        :return: an image embedding in shape of (dim,).
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)
        feats = self.model.forward_features(data)
        emb = self.post_proc(feats).squeeze(0).detach().numpy()

        return np.array(emb).astype("float32")

    def post_proc(self, features):
        features = features.to("cpu")
        if features.dim() == 3:
            features = features[:, 0]
        if features.dim() == 4:
            global_pool = torch.nn.AdaptiveAvgPool2d(1)
            features = global_pool(features)
            features = features.flatten(1)
        assert features.dim() == 2, f"Invalid output dim {features.dim()}"
        return features

    def preprocess(self, image):
        """Transform image from PIL.Image to torch.tensor with model transformations.

        :param image: image data.
        :type data: PIL.Image

        :return: an image tensor (without batch size).
        """
        from timm.data import create_transform, resolve_data_config  # pylint: disable=C0415


        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        transform = create_transform(**data_cfg)
        image_tensor = transform(image)
        return image_tensor



    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            input_size = self.model.pretrained_cfg["input_size"]
            dummy_input = torch.rand((1,) + input_size)
            feats = self.to_embeddings(dummy_input)
            self.__dimension = feats.shape[0]
        return self.__dimension
