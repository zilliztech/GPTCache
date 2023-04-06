import unittest
from langchain.llms import OpenAI
from langchain import HuggingFaceHub


class TestLangChain(unittest.TestCase):
    def test_langchain_adapter_OpenAI(self):
        question = "who are you"
        langchain_openai = OpenAI(model_name="text-ada-001")
        answer = langchain_openai(question)
        print(answer)
        # should has I am
        assert((answer.find("I am") != -1))

    def test_langchain_adapterguugingface(self):
        llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length": 64})
        question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
        answer = llm(question)
        print(answer)
        assert((answer.find("New England Patriots") != -1))


if __name__ == "__main__":
    unittest.main()