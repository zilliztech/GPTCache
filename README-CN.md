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

注：可以通过下面指令快速体验这个缓存，值得注意的是或许这不是很稳定。

```bash
pip install -i https://test.pypi.org/simple/ gpt-cache==0.0.1
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

在实际生产中，或者有一定用户群里，需要更多的考虑向量搜索这部分，可以了解下 [Milvus](https://github.com/milvus-io/milvus)，当然也有 [Milvus云服务](https://cloud.zilliz.com/) ，快速体验 Milvus 向量检索

更多参考文档：

- [更多案例](example/example.md)
- [系统设计](doc/system-cn.md)

## 🙏 感谢

感谢[ 公司 Zilliz ](https://zilliz.com/)中的同事给予我想法上的灵感和技术上的支持。