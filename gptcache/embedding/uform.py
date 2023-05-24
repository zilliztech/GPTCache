from typing import Union, Any

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import import_uform, import_pillow
from gptcache.utils.error import ParamError

import_pillow()
import_uform()

from uform import TritonClient, get_model  # pylint: disable=C0413 # nopep8
from PIL import Image  # pylint: disable=C0413 # nopep8


class UForm(BaseEmbedding):
    """Generate multi-modal embeddings using pretrained models from UForm.

    :param model: model name, defaults to 'unum-cloud/uform-vl-english'.
    :type model: str
    :param embedding_type: type of embedding, defaults to 'text'. options: text, image
    :type embedding_type: str

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

    def __init__(self, model: Union[str, TritonClient] = "unum-cloud/uform-vl-english", embedding_type: str = "text"):
        if isinstance(model, str):
            self.__model = get_model(model)
        else:
            self.__model = model
        self.__embedding_type = embedding_type
        if embedding_type == "text":
            self.__dimension = self.__model.text_encoder.proj.out_features
        elif embedding_type == "image":
            self.__dimension = self.__model.img_encoder.proj.out_features
        else:
            raise ParamError(f"Unknown embedding type: {embedding_type}")

    def to_embeddings(self, data: Any, **_):
        """Generate embedding given text input or a path to a file.

        :param data: text in string, or a path to an image file.
        :type data: str

        :return: an embedding in shape of (dim,).
        """
        if self.__embedding_type == "image":
            data = Image.open(data)
            data = self.__model.preprocess_image(data)
            emb = self.__model.encode_image(data)
        else:
            data = self.__model.preprocess_text(data)
            emb = self.__model.encode_text(data)
        return emb.detach().numpy().flatten()

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
