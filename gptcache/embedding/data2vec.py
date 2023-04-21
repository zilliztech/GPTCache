import numpy as np

from gptcache.utils import import_huggingface, import_torch, import_torchaudio
from gptcache.embedding.base import BaseEmbedding

import_torch()
import_huggingface()
import_torchaudio()

import torch  # pylint: disable=C0413
import torchaudio  # pylint: disable=C0413
from transformers import Data2VecAudioModel, Wav2Vec2Processor  # pylint: disable=C0413


class Data2VecAudio(BaseEmbedding):
    """Generate audio embedding for given audio using pretrained models from Data2Vec.

    :param model: model name, defaults to 'facebook/data2vec-audio-base-960h'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import Data2VecAudio

            audio_file = 'test.wav'
            encoder = Data2VecAudio(model='facebook/data2vec-audio-base-960h')
            embed = encoder.to_embeddings(audio_file)
    """
    def __init__(self, model_name = "facebook/data2vec-audio-base-960h"):
        self.model = Data2VecAudioModel.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.__dimension = self.model.config.hidden_size
        self.sr = self.processor.feature_extractor.sampling_rate

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: path to audio file.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        audio = self.load_audio(data, self.sr)
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        feat = last_hidden_states[:,-1,:].flatten().detach().cpu().numpy()
        return np.array(feat).astype("float32")

    def load_audio(self, audio_path, target_sr):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torch.mean(waveform, axis=0)
        transform = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = transform(waveform)
        return waveform


    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
