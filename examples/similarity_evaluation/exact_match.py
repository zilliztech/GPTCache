from gptcache.adapter import openai
from gptcache import cache
from gptcache.similarity_evaluation.exact_match import AbsoluteEvaluation


def run():
    cache.init(similarity_evaluation=AbsoluteEvaluation())
    cache.set_openai_key()

    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': 'what is chatgpt'}
        ],
    )
    print(answer)


if __name__ == '__main__':
    run()
