from gptcache import cache
from gptcache.session import Session
from gptcache.adapter import openai

# init gptcache
cache.init()
cache.set_openai_key()


def run_session():
    session = Session()
    response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                    {
                        "role": "user",
                        "content": "what's github?"
                    }],
                  session=session
                )
    response_content = response["choices"][0]["message"]["content"]
    print(response_content)


def run_custom_session():
    def my_check_hit(cur_session_id, cache_session_ids, cache_questions, cache_answer):
        print(cur_session_id, cache_session_ids, cache_questions, cache_answer)
        if "GitHub" in cache_answer:
            return True
        return False
    session = Session(name="my-session", check_hit_func=my_check_hit)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "what's github?"
            }],
        session=session
    )
    response_content = response["choices"][0]["message"]["content"]
    print(response_content)
