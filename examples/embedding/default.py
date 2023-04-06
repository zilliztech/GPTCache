from gptcache.adapter import openai
from gptcache.core import cache
from gptcache.embedding.string import to_embeddings as string_embedding


def run():
    cache.init(embedding_func=string_embedding)
    cache.set_openai_key()

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "what's chatgpt"}
        ],
    )
    print(answer)


if __name__ == '__main__':
    run()
