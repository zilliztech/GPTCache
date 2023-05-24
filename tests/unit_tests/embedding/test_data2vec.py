from io import BytesIO

import requests

from gptcache.adapter.api import _get_model
from gptcache.embedding import Data2VecAudio


def test_data2vec_audio():
    url = "https://github.com/towhee-io/examples/releases/download/data/ah_yes.wav"
    req = requests.get(url)
    audio = BytesIO(req.content) 
    t = Data2VecAudio(model="facebook/data2vec-audio-base-960h")
    data = t.to_embeddings(audio)
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    req = requests.get(url)
    audio = BytesIO(req.content) 
    t = _get_model("data2vecaudio")
    data = t.to_embeddings(audio)
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"


if __name__ == "__main__":
    test_data2vec_audio()
