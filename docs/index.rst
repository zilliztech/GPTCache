.. GPTCache documentation master file, created by
   sphinx-quickstart on Tue Apr  4 12:07:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GPTCache: A Library for Creating Semantic Cache for LLM Queries
===========================================================================

Boost LLM API Speed by 100x ⚡, Slash Costs by 10x 💰

----

.. image:: https://img.shields.io/pypi/v/gptcache?label=Release&color
   :width: 100
   :alt: release
   :target: https://pypi.org/project/gptcache/

.. image:: https://img.shields.io/pypi/dm/gptcache.svg?color=bright-green
   :width: 100
   :alt: pip_downloads
   :target: https://pypi.org/project/gptcache/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :width: 100
   :alt: License
   :target: https://opensource.org/license/mit/

.. image:: https://dcbadge.vercel.app/api/server/Q8C6WEjSWV?compact=true&style=flat
  :width: 100
  :alt: discord
  :target: https://discord.gg/Q8C6WEjSWV

.. image:: https://img.shields.io/twitter/url/https/twitter.com/zilliz_universe.svg?style=social&label=Follow%20%40Zilliz
   :width: 100
   :alt: Twitter
   :target: https://twitter.com/zilliz_universe

Quick Install
--------------

``pip install gptcache``

What is GPTCache?
--------------------

ChatGPT and various large language models (LLMs) boast incredible versatility, enabling the development of a wide range of applications. However, as your application grows in popularity and encounters higher traffic levels, the expenses related to LLM API calls can become substantial. Additionally, LLM services might exhibit slow response times, especially when dealing with a significant number of requests.

To tackle this challenge, we have created GPTCache, a project dedicated to building a semantic cache for storing LLM responses. 


Getting Started
--------------------

**Note**:

- You can quickly try GPTCache and put it into a production environment without heavy development. However, please note that the repository is still under heavy development.
- By default, only a limited number of libraries are installed to support the basic cache functionalities. When you need to use additional features, the related libraries will be **automatically installed**.
- Make sure that the Python version is **3.8.1 or higher**, check: ``python --version``
- If you encounter issues installing a library due to a low pip version, run: ``python -m pip install --upgrade pip``.

dev install
````````````
::
 

    # clone GPTCache repo
    git clone https://github.com/zilliztech/GPTCache.git
    cd GPTCache

    # install the repo
    pip install -r requirements.txt
    python setup.py install


example usage
``````````````

These examples will help you understand how to use exact and similar matching with caching. 

Before running the example, **make sure** the OPENAI_API_KEY environment variable is set by executing ``echo $OPENAI_API_KEY``. 

If it is not already set, it can be set by using ``export OPENAI_API_KEY=YOUR_API_KEY`` on Unix/Linux/MacOS systems or `set OPENAI_API_KEY=YOUR_API_KEY` on Windows systems. 

> It is important to note that this method is only effective temporarily, so if you want a permanent effect, you'll need to modify the environment variable configuration file. For instance, on a Mac, you can modify the file located at ``/etc/profile``.

To use GPTCache exclusively, only the following lines of code are required, and there is no need to modify any existing code.

::

    from gptcache import cache
    from gptcache.adapter import openai

    cache.init()
    cache.set_openai_key()


More Docs:

- `Usage, how to use GPTCache better <./usage.html>`_
- `Features, all features currently supported by the cache <./feature.html>`_
- `Examples, learn better custom caching <https://github.com/zilliztech/GPTCache/tree/main/examples>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting-started
   :hidden:

   usage.md
   feature.md

What can this help with?
-------------------------

GPTCache offers the following primary benefits:

- **Decreased expenses**: Most LLM services charge fees based on a combination of number of requests and `token count <https://openai.com/pricing>`_. By caching query results, GPTCache reduces both the number of requests and the number of tokens sent to the LLM service. This minimizes the overall cost of using the service. 
- **Enhanced performance**: LLMs employ generative AI algorithms to generate responses in real-time, a process that can sometimes be time-consuming. However, when a similar query is cached, the response time significantly improves, as the result is fetched directly from the cache, eliminating the need to interact with the LLM service. In most situations, GPTCache can also provide superior query throughput compared to standard LLM services.
- **Improved scalability and availability**: LLM services frequently enforce `rate limits <https://platform.openai.com/docs/guides/rate-limits>`_, which are constraints that APIs place on the number of times a user or client can access the server within a given timeframe. Hitting a rate limit means that additional requests will be blocked until a certain period has elapsed, leading to a service outage. With GPTCache, you can easily scale to accommodate an increasing volume of of queries, ensuring consistent performance as your application's user base expands.
- **Flexible development environment**: When developing LLM applications, an LLM APIs connection is required to prove concepts. GPTCache offers the same interface as LLM APIs and can store LLM-generated or mocked data. This helps to verify your application's features without connecting to the LLM APIs or even the network.

How does it work?
------------------

Online services often exhibit data locality, with users frequently accessing popular or trending content. Cache systems take advantage of this behavior by storing commonly accessed data, which in turn reduces data retrieval time, improves response times, and eases the burden on backend servers. Traditional cache systems typically utilize an exact match between a new query and a cached query to determine if the requested content is available in the cache before fetching the data.

However, using an exact match approach for LLM caches is less effective due to the complexity and variability of LLM queries, resulting in a low cache hit rate. To address this issue, GPTCache adopt alternative strategies like semantic caching. Semantic caching identifies and stores similar or related queries, thereby increasing cache hit probability and enhancing overall caching efficiency. 

GPTCache employs embedding algorithms to convert queries into embeddings and uses a vector store for similarity search on these embeddings. This process allows GPTCache to identify and retrieve similar or related queries from the cache storage, as illustrated in the `Modules section <https://github.com/zilliztech/GPTCache#-modules>`_. 

Featuring a modular design, GPTCache makes it easy for users to customize their own semantic cache. The system offers various implementations for each module, and users can even develop their own implementations to suit their specific needs.

In a semantic cache, false positives can occur during cache hits and false negatives during cache misses. GPTCache provides three metrics to evaluate its performance:

- Precision: the ratio of true positives to the total of true positives and false positives.
- Recall: the ratio of true positives to the total of true positives and false negatives.
- Latency: the time required for a query to be processed and the corresponding data to be fetched from the cache.

A `sample benchmark <https://github.com/zilliztech/gpt-cache/blob/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py>`_ is included for users to start with assessing the performance of their semantic cache.



Modules
----------

.. image:: ./GPTCacheStructure.png

You can take a look at modules below to learn more about system design and architecture.

- `LLM Adapter <modules/llm_adapter.html>`_
- `Embedding Generator <modules/embedding_generator.html>`_
- `Cache Storage <modules/cache_storage.html>`_
- `Vector Store <modules/vector_store.html>`_
- `Cache Manager <modules/cache_manager.html>`_
- `Similarity Evaluator <modules/similarity_evaluator.html>`_

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

**Note**:
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


Roadmap
-------

Coming soon! `Stay tuned! <https://twitter.com/zilliz_universe>`_


Contributing
---------------

WWe are extremely open to contributions, be it through new features, enhanced infrastructure, or improved documentation.

For comprehensive instructions on how to contribute, please refer to our `contribution guide <contributing.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :name: contributing
   :hidden:

   contributing.md
