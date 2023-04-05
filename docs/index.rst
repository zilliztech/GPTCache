.. GPT Cache documentation master file, created by
   sphinx-quickstart on Tue Apr  4 12:07:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPT Cache!
=====================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   module
   references/index
   markdown/index

🤠 What is GPT Cache?
---------------------

.. image:: https://dcbadge.vercel.app/api/server/Q8C6WEjSWV?compact=true&style=flat
  :width: 100
  :alt: discord
  :target: https://discord.gg/Q8C6WEjSWV

Large Language Models (LLMs) are a promising and transformative technology that has rapidly advanced in recent years. These models are capable of generating natural language text and have numerous applications, including chatbots, language translation, and creative writing. However, as the size of these models increases, so do the costs and performance requirements needed to utilize them effectively. This has led to significant challenges in developing on top of large models such as ChatGPT.

To address this issue, we have developed GPT Cache, a project that focuses on caching responses from language models, also known as a semantic cache. The system offers two major benefits:

1. Quick response to user requests: the caching system provides faster response times compared to large model inference, resulting in lower latency and faster response to user requests.
2. Reduced service costs: most LLM services are currently charged based on the number of tokens. If user requests hit the cache, it can reduce the number of requests and lower service costs.

🤔 Why would GPT Cache be helpful?
----------------------------------

A good analogy for GptCache is to think of it as a more semantic version of Redis. In GptCache, hits are not limited to exact matches, but rather also include prompts and context similar to previous queries. We believe that the traditional cache design still works for AIGC applications for the following reasons:

- Locality is present everywhere. Like traditional application systems, AIGC applications also face similar hot topics. For instance, ChatGPT itself may be a popular topic among programmers.
- For purpose-built SaaS services, users tend to ask questions within a specific domain, with both temporal and spatial locality.
- By utilizing vector similarity search, it is possible to find a similarity relationship between questions and answers at a relatively low cost.

We provide `benchmark <https://github.com/zilliztech/gpt-cache/blob/main/examples/benchmark/benchmark_sqlite_faiss_towhee.py>`_ to illustrate the concept. In semantic caching, there are three key measurement dimensions: false positives, false negatives, and hit latency. With the plugin-style implementation, users can easily tradeoff these three measurements according to their needs.

😆 Contributing
===============

Would you like to contribute to the development of GPT Cache? Take a look at `our contribution guidelines <./contributing.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :name: contributing
   :hidden:

   contributing.md
