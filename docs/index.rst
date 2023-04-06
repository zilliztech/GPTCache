.. GPT Cache documentation master file, created by
   sphinx-quickstart on Tue Apr  4 12:07:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPT Cache!
=====================================
----
.. image:: https://dcbadge.vercel.app/api/server/Q8C6WEjSWV?compact=true&style=flat
  :width: 100
  :alt: discord
  :target: https://discord.gg/Q8C6WEjSWV

.. image:: https://img.shields.io/pypi/v/gptcache?label=Release&color
   :width: 100
   :alt: release
   :target: https://pypi.org/project/gptcache/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :width: 100
   :alt: License
   :target: https://opensource.org/license/mit/
|

Large Language Models (LLMs) are a promising and transformative technology that has rapidly advanced in recent years. These models are capable of generating natural language text and have numerous applications, including chatbots, language translation, and creative writing. However, as the size of these models increases, so do the costs and performance requirements needed to utilize them effectively. This has led to significant challenges in developing on top of large models such as ChatGPT.

To address this issue, we have developed **GPT Cache**, a project that focuses on caching responses from language models, also known as a semantic cache. The system offers two major benefits:

- **Quick response to user requests:** the caching system provides faster response times compared to large model inference, resulting in lower latency and faster response to user requests.
- **Reduced service costs:** most LLM services are currently charged based on the number of tokens. If user requests hit the cache, it can reduce the number of requests and lower service costs.

To learn more details about GPTCache and follow up updates, please visit our `README.md <https://github.com/zilliztech/GPTCache/blob/main/README.md>`_.


Getting Started
---------------

Check out `the guide <./quick-start.html>`_ to get started with **GPTCache**.
You can follow instructions to simply install and run an example at local.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: getting-started
   :hidden:

   quick-start


Overview
----------
A good analogy for GptCache is to think of it as a more semantic version of Redis. In GptCache, hits are not limited to exact matches, but rather also include prompts and context similar to previous queries. We believe that the traditional cache design still works for AIGC applications for the following reasons:

- Locality is present everywhere. Like traditional application systems, AIGC applications also face similar hot topics. For instance, ChatGPT itself may be a popular topic among programmers.
- For purpose-built SaaS services, users tend to ask questions within a specific domain, with both temporal and spatial locality.
- By utilizing vector similarity search, it is possible to find a similarity relationship between questions and answers at a relatively low cost.

We provide `benchmark <https://github.com/zilliztech/GPTCache/blob/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py>`_ to illustrate the concept. In semantic caching, there are three key measurement dimensions: false positives, false negatives, and hit latency. With the plugin-style implementation, users can easily tradeoff these three measurements according to their needs.

You can take a look at `system architecture <./system.html>`_ and `modules <./module.html>`_ to learn about GPTCache design and architecture.

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :name: overview
   :hidden:

   system
   module


.. Examples
.. -----------


References
----------------

For more information about API and examples, you can checkout `API References <./references/index.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: References
   :name: references
   :hidden:

   references/index


Contributing
---------------

Would you like to contribute to the development of GPT Cache? Take a look at `our contribution guidelines <./contributing.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :name: contributing
   :hidden:

   contributing
