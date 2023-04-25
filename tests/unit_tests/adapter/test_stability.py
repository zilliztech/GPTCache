from unittest.mock import patch
import base64
from io import BytesIO
import os
import numpy as np


from gptcache.adapter import stability_sdk as cache_stability
from gptcache.adapter.stability_sdk import generation, construct_resp_from_cache
from gptcache import cache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from gptcache.utils import (
    import_stability, import_pillow
    )

import_pillow()
import_stability()

import stability_sdk
from PIL import ImageChops, Image


def test_stability_inference_map():
    cache.init(pre_embedding_func=get_prompt)
    expected_img = Image.new("RGB", (1, 1))

    buffered = BytesIO()
    expected_img.save(buffered, format="PNG")
    test_img_b64 = base64.b64encode(buffered.getvalue())
    expected_response = construct_resp_from_cache(test_img_b64, 1, 1)

    stability_api = cache_stability.StabilityInference(key="ThisIsTest")
    with patch.object(stability_sdk.client.StabilityInference, "generate") as mock_call:
        mock_call.return_value = expected_response

        answer_response = stability_api.generate(prompt="Test prompt", width=1, height=1)
        answers = []
        for resp in answer_response:
            for artifact in resp.artifacts:      
                if artifact.type == generation.ARTIFACT_IMAGE:
                    answers.append(Image.open(BytesIO(artifact.binary)))
        assert len(answers) == 1, f"Expect to get 1 image but got {len(answers)}"
        diff = ImageChops.difference(answers[0], expected_img)
        assert not diff.getbbox()

    answer_response = stability_api.generate(prompt="Test prompt", width=2, height=2)
    answers = []
    for resp in answer_response:
        for artifact in resp.artifacts:        
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(BytesIO(artifact.binary))
                assert img.size == (2, 2), "Incorrect image size."
                answers.append(img)
    assert len(answers) == 1, f"Expect to get 1 image but got {len(answers)}"
    diff = ImageChops.difference(answers[0], expected_img)
    assert not diff.getbbox()


def test_stability_inference_faiss():
    faiss_file = "faiss.index"
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

    data_manager = manager_factory("sqlite,faiss,local", 
                               data_dir="./", 
                               vector_params={"dimension": 2},
                               object_params={"path": "./images"}
                               )
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=lambda x, **_: np.ones((2,)).astype("float32"),
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation()
    )

    expected_img = Image.new("RGB", (1, 1))

    buffered = BytesIO()
    expected_img.save(buffered, format="PNG")
    test_img_b64 = base64.b64encode(buffered.getvalue())
    expected_response = construct_resp_from_cache(test_img_b64, 1, 1)

    with patch("stability_sdk.client.StabilityInference.generate") as mock_call:
        mock_call.return_value = expected_response

        stability_api = cache_stability.StabilityInference(key="ThisIsTest")
        answer_response = stability_api.generate(prompt="Test prompt", width=1, height=1)
        answers = []
        for resp in answer_response:
            for artifact in resp.artifacts:        
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(BytesIO(artifact.binary))
                    assert img.size == (1, 1), "Incorrect image size."
                    answers.append(img)
        assert len(answers) == 1, f"Expect to get 1 image but got {len(answers)}"
        diff = ImageChops.difference(answers[0], expected_img)
        assert not diff.getbbox()

    answer_response = stability_api.generate(prompt="Test prompt", width=2, height=2)
    answers = []
    for resp in answer_response:
        for artifact in resp.artifacts:        
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(BytesIO(artifact.binary))
                assert img.size == (2, 2), "Incorrect image size."
                answers.append(img)
    assert len(answers) == 1, f"Expect to get 1 image but got {len(answers)}"
    diff = ImageChops.difference(answers[0], expected_img)
    assert not diff.getbbox()



if __name__ == "__main__":
    test_stability_inference_map()
    test_stability_inference_faiss()
        

    