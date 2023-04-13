import os

from langchain import Cohere
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache import cache
from gptcache.processor.pre import get_prompt

from gptcache.adapter.langchain_models import LangChainChat

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
Cohere.cohere_api_key = os.getenv("COHERE_API_KEY")


def run_llm():
    cache.init(
        pre_embedding_func=get_prompt,
    )

    question = 'what is chatgpt'

    langchain_openai = OpenAI(model_name='text-ada-001')
    llm = LangChainLLMs(llm=langchain_openai)
    answer = llm(question)
    print(answer)

    # TODO install cohere auto
    langchain_cohere = Cohere()
    llm = LangChainLLMs(llm=langchain_cohere)
    answer = llm(question)
    print(answer)


def get_msg(data, **_):
    return data.get("messages")[-1].content


def run_chat_model():
    cache.init(
        pre_embedding_func=get_msg,
    )

    # chat=ChatOpenAI(temperature=0)
    chat = LangChainChat(chat=ChatOpenAI(temperature=0))

    answer = chat([HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")])
    print(answer)


if __name__ == '__main__':
    run_llm()
    run_chat_model()
