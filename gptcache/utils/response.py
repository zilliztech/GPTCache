def get_message_from_openai_answer(openai_resp):
    return openai_resp["choices"][0]["message"]["content"]


def get_stream_message_from_openai_answer(openai_data):
    return openai_data["choices"][0]["delta"].get("content", "")


def get_text_from_openai_answer(openai_resp):
    return openai_resp["choices"][0]["text"]
