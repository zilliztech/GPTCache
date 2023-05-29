import pytest

from gptcache.adapter.api import _get_pre_context_function
from gptcache.utils import import_selective_context


@pytest.mark.tags("L2")
def test_selective_context_process():
    import_selective_context()
    context_process = _get_pre_context_function("selective")
    chat = []
    chat.append(
        {
            "role": "user",
            "content": "What is the meaning of life? Explain it in 5 paragraphs.",
        }
    )
    chat.append(
        {
            "role": "system",
            "content": "The meaning of life is a philosophical question that has been debated throughout history. It is a complex and multifaceted topic, and different people may have different answers depending on their beliefs, values, and experiences. In general, though, the meaning of life is thought to be the reason for which we exist, the purpose that gives our lives significance and direction. \
One possible answer to the question of the meaning of life is that it is to seek happiness and fulfillment. This is a common theme in many philosophical and religious traditions, which often emphasize the importance of living a good and virtuous life in order to achieve happiness and fulfillment. For some, this may mean striving for personal growth and self-improvement, while for others it may involve pursuing spiritual enlightenment or a connection with a higher power. \
Another possible answer to the question of the meaning of life is that it is to contribute to the greater good. This approach emphasizes the importance of serving others and making the world a better place, whether through acts of kindness, charitable work, or other forms of service. For those who believe in this approach, the meaning of life is not just about individual happiness, but also about making a positive impact on the world and the lives of others. \
A third possible answer to the question of the meaning of life is that it is to discover and pursue one's passions and interests. This approach emphasizes the importance of living a life that is authentic and true to oneself, and of pursuing one's passions and interests with enthusiasm and dedication. For those who believe in this approach, the meaning of life is not just about achieving some external goal, but about living a life that is rich, fulfilling, and true to who we are. \
In conclusion, the meaning of life is a deeply personal and subjective question, and different people may have different answers. Some may believe that the meaning of life is to seek happiness and fulfillment, while others may believe that it is to contribute to the greater good or to pursue one's passions and interests. Ultimately, the meaning of life is a mystery that may never be fully understood, but that we can strive to uncover and understand through our experiences, beliefs, and values.",
        }
    )
    chat.append({"role": "user", "content": "Shouldn't the answer be 42?"})
    chat.append(
        {
            "role": "system",
            "content": """The answer "42" is a reference to a famous line in the book "The Hitchhiker's Guide to the Galaxy" by Douglas Adams. In the book, the character Deep Thought is asked to find the answer to the ultimate question of life, the universe, and everything. After thinking for a very long time, Deep Thought reveals that the answer is "42." However, the characters in the book then realize that they don't actually know what the question is, so the answer is ultimately meaningless. \
In other words, the answer "42" is a humorous and satirical take on the idea that there may be a single, definitive answer to the question of the meaning of life. It suggests that the search for such an answer may be futile, and that the meaning of life may be something that is ultimately unknowable. Therefore, while "42" is a famous and memorable line from a popular book, it is not necessarily a serious or meaningful answer to the question of the meaning of life.""",
        }
    )

    context_process.format_all_content({"messages": chat})
    save_content, embedding_content = context_process.process_all_content()
    assert len(save_content) > len(embedding_content)
