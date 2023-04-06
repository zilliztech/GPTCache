.. GPTCache documentation master file, created by
   sphinx-quickstart on Tue Apr  4 12:07:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPTCache!
=====================================

Boost LLM API Speed by 100x ⚡, Slash Costs by 10x 💰

.. image:: https://img.shields.io/pypi/v/gptcache?label=Release&color
   :width: 100
   :alt: release
   :target: https://pypi.org/project/gptcache/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :width: 100
   :alt: License
   :target: https://opensource.org/license/mit/

.. image:: https://dcbadge.vercel.app/api/server/Q8C6WEjSWV?compact=true&style=flat
  :width: 100
  :alt: discord
  :target: https://discord.gg/Q8C6WEjSWV

`Discord <https://discord.gg/Q8C6WEjSWV>`_ | `Twitter <https://twitter.com/zilliz_universe>`_


Large Language Models (LLMs) are a promising and transformative technology that has rapidly advanced in recent years. These models are capable of generating natural language text and have numerous applications, including chatbots, language translation, and creative writing. However, as the size of these models increases, so do the costs and performance requirements needed to utilize them effectively. This has led to significant challenges in developing on top of large models such as ChatGPT.

To address this issue, we have developed **GPTCache**, a project that focuses on caching responses from language models, also known as a semantic cache. The system offers two major benefits:

- **Quick response to user requests:** the caching system provides faster response times compared to large model inference, resulting in lower latency and faster response to user requests.
- **Reduced service costs:** most LLM services are currently charged based on the number of tokens. If user requests hit the cache, it can reduce the number of requests and lower service costs.

To learn more details about GPTCache and follow the updates, please visit our `README.md <https://github.com/zilliztech/GPTCache/blob/main/README.md>`_.


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


Modules
----------

.. image:: ./GPTCacheStructure.png

You can take a look at modules below to learn more about system design and architecture.

- `LLM Adapter <modules/llm_adapter.html>`_
- `Embedding Generator <modules/embedding_generator.html>`_
- `Cache Storage <modules/cache_storage.html>`_
- `Vector Store <modules/vector_store>`_
- `Cache Manager <modules/cache_manager>`_
- `Similarity Evaluator <modules/similarity_evaluator>`_

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :name: modules
   :hidden:

   modules/llm_adapter
   modules/embedding_generator
   modules/cache_storage
   modules/vector_store
   modules/cache_manager
   modules/similarity_evaluator

**Note:**
Not all combinations of different modules may be compatible with each other. For instance, if we disable the **Embedding Extractor**, the **Vector Store** may not function as intended. We are currently working on implementing a combination sanity check for **GPTCache**.


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

Would you like to contribute to the development of GPTCache? Take a look at `our contribution guidelines <./contributing.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :name: contributing
   :hidden:

   contributing
