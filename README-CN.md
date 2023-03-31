# GPTCache

[English](README-CN.md) | 中文

GPT Cache主要用于缓存用户在使用ChatGPT的问答数据。这个系统带来两个好处：

1. 快速响应用户请求：相比于大模型推理，缓存系统中查找数据将具有更低的延迟，从而更快地响应用户请求。
2. 降低服务成本：目前大多数ChatGPT服务都是基于请求次数进行收费，如果用户请求命中缓存，就可以减少请求次数，从而降低服务成本。

如果这个想法💡对你很有帮助，帮忙给个star 🌟，甚是感谢！

## 🤔 是否有必要使用缓存？

我认为有必要，理由如下：

- 基于ChatGPT开发的某些领域服务，许多问答具有一定的相似性。
- 对于一个用户，使用ChatGPT提出的一系列问题具有一定规律性，与其职业、生活习惯、性格等有一定关联。例如，程序员使用ChatGPT服务的可能性很大程度上与其工作有关。
- 如果您提供的ChatGPT服务面向大量用户群体，将其分为不同的类别，那么相同类别中的用户问的相关问题也有很大概率命中缓存，从而降低服务成本。

## 😊 快速接入

### alpha 测试包安装

注：
- 可以通过下面指令快速体验这个缓存，值得注意的是或许这不是很稳定。
- 默认情况下，基本上不需要安装什么第三方库。当需要使用一些特性的时候，使用前需要自己进行安装，参考：[安装依赖列表](doc/installation.md)。例如，如果想使用milvus做为向量存储，应该安装pymilvus，即：
```
pip install pymilvus
```

```bash
# create conda new environment
conda create --name gpt-cache python=3.8
conda activate gpt-cache

# clone gpt cache repo
git clone https://github.com/zilliztech/gpt-cache
cd gpt-cache

# install the repo
pip install -r requirements.txt
python setup.py install
```

如果只是想实现请求的精准匹配缓存，即两次一模一样的请求，则只需要**两步**就可以接入这个cache !!!

1. cache初始化
```python
from gpt_cache.core import cache
cache.init()
# 如果使用`openai.api_key = xxx`设置API KEY，需要用下面语句替换它
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

在本地运行，如果想要更好的效果，可以使用示例中的 [Sqlite + Faiss + Towhee](example/sf_towhee/sf_manager.py) 方案，其中 Sqlite + Faiss 进行缓存数据管理，Towhee 进行 embedding 操作。

在实际生产中，或者有一定用户群里，需要更多的考虑向量搜索这部分，可以了解下 [Milvus](https://github.com/milvus-io/milvus)，当然也有 [Zilliz 云服务](https://cloud.zilliz.com/) ，快速体验 Milvus 向量检索

更多参考文档：

- [更多案例](example/example.md)
- [系统设计](doc/system-cn.md)

## 🥳 功能

- 支持openai普通和流式的聊天请求
- 支持top_k搜索，可以在DataManager创建时进行设置
- 支持多级缓存, 参考: `Cache#next_cache`

```python
bak_cache = Cache()
bak_cache.init()
cache.init(next_cache=bak_cache)
```

- 是否跳过当前缓存，对于请求不进行缓存搜索也不保存chat gpt返回的结果，参考： `Cache#cache_enable_func`
- 缓存系统初始化阶段，不进行缓存搜索，但是保存chat gpt返回的结果，参考： 使用`create`方法时设置`cache_skip=True`参数

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=mock_messages,
    cache_skip=True,
)
```

- 像积木一样，所有模块均可自定义，包括：
  - pre-embedding，获取原始请求中的特征信息，如最后一条消息，prompt等
  - embedding，将特征信息转换成向量数据
  - data manager，缓存数据管理，主要包括数据搜索和保存
  - cache similarity evaluation，可以使用相似搜索的距离或者其他更适合使用场景的模型
  - post-process，处理缓存答案列表，比如最相似的，随机或者自定义

## 🤗 所有模块

- Pre-embedding
  - 获取请求的最后一条消息, 参考: `pre_embedding.py#last_content`
- Embedding
  - [x] [towhee](https://towhee.io/), 英语模型: paraphrase-albert-small-v2, 中文模型: uer/albert-base-chinese-cluecorpussmall
  - [x] openai embedding api
  - [x] string, 不做任何处理
  - [ ] [cohere](https://docs.cohere.ai/reference/embed) embedding api  
- Data Manager
  - 标量存储
    - [x] [sqlite](https://sqlite.org/docs.html)
    - [ ] [postgresql](https://www.postgresql.org/)
    - [ ] [mysql](https://www.mysql.com/)
  - 向量存储
    - [x] [milvus](https://milvus.io/)
  - 向量索引
    - [x] [faiss](https://faiss.ai/)
- Similarity Evaluation
  - 搜索距离, 参考: `simple.py#pair_evaluation`
  - [towhee](https://towhee.io/), roberta_duplicate模型, 问题与问题相关性匹配，只支持512个token
  - string, 缓存问题和输入问题字符匹配
  - np, 使用`linalg.norm`进行向量距离计算
- Post Process
  - 选择最相似的答案
  - 随机选择

## 😆 贡献
想一起构建GPT Cache吗？相信[贡献文档](doc/contributing.md)可以给予你一定帮助。

## 🙏 感谢

感谢[ 公司 Zilliz ](https://zilliz.com/)中的同事给予我想法上的灵感和技术上的支持。