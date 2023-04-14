# 🎉 Introduction to new functions of GPTCache

To read the following content, you need to understand the basic use of GPTCache, references:

- [Readme doc](https://github.com/zilliztech/GPTCache)
- [Usage doc](https://github.com/zilliztech/GPTCache/blob/main/docs/usage.md)

## v0.1.10 (2023.4.13)

1. Add kreciprocal similarity evaluation

K-reprciprocl evaluation is a method inspired by the popular reranking method in ReID(https://arxiv.org/abs/1701.08398). The term “k-reciprocal” comes from the fact that the algorithm creates reciprocal relationships between similar embeddings in the top-k list. In other words, if embedding A is similar to embedding B and embedding B is similar to embedding A, then A and B are said to be “reciprocally similar” to each other. This evaluation abandon those embeddings pairs which are not “reciprocally similar” in their K nearest neighbors. And the remaining pairs will keep the distance for the final rank.

```
vector_base = VectorBase("faiss", dimension=d)
data_manager = get_data_manager(CacheBase("sqlite"), vector_base)
evaluation = KReciprocalEvaluation(vectordb=vector_base)
cache.init(
    ... # other configs
    data_manager=data_manager,
    similarity_evaluation=evaluation,
)
```

2. Add LangChainChat adapter

```
from gptcache.adapter.langchain_models import LangChainChat

cache.init(
    pre_embedding_func=get_msg,
)

chat = LangChainChat(chat=ChatOpenAI(temperature=0))
answer = chat(
    messages=[
        HumanMessage(
            content="Translate this sentence from English to Chinese. I love programming."
        )
    ]
)
```

## v0.1.9 (2023.4.12)

1. Import data into cache

```
cache.init()

questions = ["foo1", "foo2"]
answers = ["a1", "a2"]
cache.import_data(questions=questions, answers=answers)
```

2. New pre-process function: remove prompts

When using the LLM model, a prompt may be added for each input. If the entire message with the prompt is brought into the cache, it may lead to an increase in the cache error hit rate. For example, the text of the prompt is very long, and the text of the real question is very short. .

```
cache_obj.init(
    pre_embedding_func=last_content_without_prompt,
    config=Config(prompts=["foo"]),
)
```

3. Embeded milvus

The embedded Milvus is a lightweight version of Milvus that can be embedded into your Python application. It is a single binary that can be easily installed and run on your machine.

```
with TemporaryDirectory(dir="./") as root:
    db = VectorBase(
                    "milvus",
                    local_mode=True,
                    local_data=str(root),
                    ... #other config
                )
    data_manager = get_data_manager("sqlite", vector_base)
```