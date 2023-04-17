import base64
import requests


def get_message_from_openai_answer(openai_resp):
    return openai_resp["choices"][0]["message"]["content"]


def get_stream_message_from_openai_answer(openai_data):
    return openai_data["choices"][0]["delta"].get("content", "")


def get_text_from_openai_answer(openai_resp):
    return openai_resp["choices"][0]["text"]


def get_image_from_openai_b64(openai_resp):
    return openai_resp["data"][0]["b64_json"]


def get_image_from_openai_url(openai_resp):
    url = openai_resp["data"][0]["url"]
    img_content = requests.get(url).content
    img_data = base64.b64encode(img_content)
    return img_data


def get_image_from_path(openai_resp):
    img_path = openai_resp["data"][0]["url"]
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read())
    return img_data


def get_audio_text_from_openai_answer(openai_resp):
    return openai_resp["text"]
