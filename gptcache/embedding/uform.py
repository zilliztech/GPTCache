import os

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_uform, import_pillow

import_pillow()
import_uform()

from uform import get_model  # pylint: disable=C0413 # nopep8
from PIL import Image  # pylint: disable=C0413 # nopep8


class UForm(BaseEmbedding):
    """Generate multi-modal embeddings using pretrained models from UForm.

    :param model: model name, defaults to 'unum-cloud/uform-vl-english'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import UForm

            test_sentence = 'Hello, world.'
            encoder = UForm(model='unum-cloud/uform-vl-english')
            embed = encoder.to_embeddings(test_sentence)

            test_sentence = '什么是Github'
            encoder = UForm(model='unum-cloud/uform-vl-multilingual')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "unum-cloud/uform-vl-english"):
        self.model = get_model(model)
        self.__dimension = self.model.image_encoder.dim

    def to_embeddings(self, data: str, **_):
        """Generate embedding given text input or a path to a file.

        :param data: text in string, or a path to an image file.
        :type data: str

        :return: an embedding in shape of (dim,).
        """
        if os.path.exists(data):
            data = Image.open(data)
            data = self.model.preprocess_image(data)
            emb = self.model.encode_image(data)
            return emb.detach().numpy().flatten()
        else:
            data = self.model.preprocess_text(data)
            emb = self.model.encode_text(data)
            return emb.detach().numpy().flatten()

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
