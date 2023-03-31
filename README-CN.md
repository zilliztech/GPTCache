# GPTCache

[English](README-CN.md) | 中文

## 🤠 什么是GPT Cache？

大型语言模型（LLMs）是一种有前途和具有变革性的技术，近年来迅速发展。这些模型能够生成自然语言文本，并具有许多应用，包括聊天机器人、语言翻译和创意写作。然而，随着这些模型的规模增大，使用它们需要的成本和性能要求也增加了。这导致了在大型模型上开发ChatGPT等应用程序方面的重大挑战。

为了解决这个问题，我们开发了GPT Cache，这是一个专注于缓存语言模型响应的项目，也称为语义缓存。该系统提供了两个主要的好处：

1. 快速响应用户请求：缓存系统提供比大型模型推理更快的响应时间，从而降低延迟并更快地响应用户请求。
2. 降低服务成本：目前大多数ChatGPT服务是基于请求数量收费的。如果用户请求命中缓存，它可以减少请求数量并降低服务成本。

如果这个想法💡对你很有帮助，帮忙给个star 🌟，甚是感谢！

## 🤔 GPT缓存为什么会有帮助？

我认为是有必要，原因是：

- 局部性无处不在。像传统的应用系统一样，AIGC应用程序也面临类似的热点问题。例如，ChatGPT本身可能是程序员们热议的话题。
- 面向特定领域的SaaS服务，用户往往在特定的领域内提出问题，具有时间和空间上的局部性。
- 通过利用向量相似度搜索，可以以相对较低的成本找到问题和答案之间的相似关系。

我们还提供了[基准测试](https://github.com/zilliztech/gpt-cache/blob/main/example/benchmark/benchmark_sqlite_faiss_towhee.py)来说明这个概念。在语义缓存中，有三个关键的测量维度：误报率、漏报率和命中延迟。通过插件式的实现，用户可以根据自己的需求轻松权衡这三个测量值。

## 😊 快速接入

注：
- 可以通过下面指令快速体验这个缓存，值得注意的是目前项目正在开发，API可能存在变更
- 默认情况下，基本上不需要安装什么第三方库。当需要使用一些特性的时候，相关的第三方库会自动下载。
- 如果因为pip版本低安装第三方库失败，使用：`python -m pip install --upgrade pip`

### pip 安装

```bash
pip install gpt_cache
```

### dev 安装

```bash
# clone gpt cache repo
git clone https://github.com/zilliztech/gpt-cache
cd gpt-cache

# install the repo
pip install -r requirements.txt
python setup.py install
```

### 快速使用

如果只是想实现请求的精准匹配缓存，即两次一模一样的请求，则只需要**两步**就可以接入这个cache !!!

1. cache初始化
```python
from gpt_cache.core import cache
cache.init()
# 如果使用`openai.api_key = xxx`设置API KEY，需要用下面语句替换它
# 方法读取OPENAI_API_KEY环境变量并进行设置，保证key的安全性 
cache.set_openai_key()
```
2. 替换原始openai包
```python
from gpt_cache.view import openai

# openai请求不需要做任何改变
answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "foo"}
        ],
    )
```

如果想快速在本地体验下向量相似搜索缓存，参考案例：[Sqlite + Faiss + Towhee](example/sqlite_faiss_towhee/sqlite_faiss_towhee.py)

更多参考文档：

- [系统设计，了解系统如何被构建](doc/system-cn.md)
- [功能，当前支持的所有特性](doc/feature_cn.md)
- [案例，更加了解如何定制化缓存](example/example.md)

## 🤗 所有模块

![GPTCache Struct](doc/GPTCacheStructure.png)

- Pre-embedding，提取请求中的关键信息
  - 获取请求的最后一条消息, 参考: `pre_embedding.py#last_content`
- Embedding，将文本转换为向量，后续进行相似搜索
  - [x] [towhee](https://towhee.io/), 英语模型: paraphrase-albert-small-v2, 中文模型: uer/albert-base-chinese-cluecorpussmall
  - [x] openai embedding api
  - [x] string, 不做任何处理
  - [ ] [cohere](https://docs.cohere.ai/reference/embed) embedding api  
- Cache，缓存数据管理，包括搜索、存储和清理
  - 标量存储
    - [x] [sqlite](https://sqlite.org/docs.html)
    - [ ] [postgresql](https://www.postgresql.org/)
    - [ ] [mysql](https://www.mysql.com/)
  - 向量存储
    - [x] [milvus](https://milvus.io/)
  - 向量索引
    - [x] [faiss](https://faiss.ai/)
- Similarity Evaluation，评估缓存结果
  - 搜索距离, 参考: `simple.py#pair_evaluation`
  - [towhee](https://towhee.io/), albert_duplicate模型, 问题与问题相关性匹配，只支持512个token
  - string, 缓存问题和输入问题字符匹配
  - np, 使用`linalg.norm`进行向量距离计算
- Post Process，如何将多个缓存答案返回给用户
  - 选择最相似的答案
  - 随机选择

## 😆 贡献
想要帮助建设GPT缓存吗？请查看我们的[贡献指南](doc/contributing.md)。

## 🙏 感谢

感谢[ 公司 Zilliz ](https://zilliz.com/)中的同事给予我想法上的灵感和技术上的支持。
