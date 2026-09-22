import atexit
import os
from typing import Optional, List, Any

from gptcache.config import Config
from gptcache.embedding.string import to_embeddings as string_embedding
from gptcache.manager import get_data_manager
from gptcache.manager.data_manager import DataManager
from gptcache.processor.exact_match import ExactMatchCache
from gptcache.processor.post import temperature_softmax
from gptcache.processor.pre import last_content
from gptcache.report import Report
from gptcache.similarity_evaluation import ExactMatchEvaluation
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils import import_openai, import_anthropic
from gptcache.utils.cache_func import cache_all
from gptcache.utils.log import gptcache_log


def _pin_faiss_threads():
    """Pin FAISS OpenMP thread count for deterministic tail latency.

    Order of precedence:
      1. GPTCACHE_FAISS_THREADS env var (explicit override)
      2. OMP_NUM_THREADS env var (respect any existing setting)
      3. min(os.cpu_count(), 4) - sensible default that avoids over-subscription

    Silently no-ops if FAISS isn't installed.
    """
    n = os.environ.get("GPTCACHE_FAISS_THREADS")
    if n is None:
        n = os.environ.get("OMP_NUM_THREADS")
    if n is None:
        n = min(os.cpu_count() or 1, 4)
    try:
        n = max(int(n), 1)
    except (TypeError, ValueError):
        return
    try:
        import faiss  # pylint: disable=C0415
        faiss.omp_set_num_threads(n)
    except ImportError:
        pass


class Cache:
    """GPTCache core object.


    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import openai

            cache.init()
            cache.set_openai_key()
    """

    # it should be called when start the cache system
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager: Optional[DataManager] = None
        self.similarity_evaluation: Optional[SimilarityEvaluation] = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report()
        self.next_cache = None
        self.exact_match_cache: Optional[ExactMatchCache] = None

    def init(
        self,
        cache_enable_func=cache_all,
        pre_embedding_func=last_content,
        pre_func=None,
        embedding_func=string_embedding,
        data_manager: DataManager = get_data_manager(),
        similarity_evaluation=ExactMatchEvaluation(),
        post_process_messages_func=temperature_softmax,
        post_func=None,
        config=Config(),
        next_cache=None,
    ):
        """Pass parameters to initialize GPTCache.

        :param cache_enable_func: a function to enable cache, defaults to ``cache_all``
        :param pre_embedding_func: a function to preprocess embedding, defaults to ``last_content``
        :param pre_func: a function to preprocess embedding, same as ``pre_embedding_func``
        :param embedding_func: a function to extract embeddings from requests for similarity search, defaults to ``string_embedding``
        :param data_manager: a ``DataManager`` module, defaults to ``get_data_manager()``
        :param similarity_evaluation: a module to calculate embedding similarity, defaults to ``ExactMatchEvaluation()``
        :param post_process_messages_func: a function to post-process messages, defaults to ``temperature_softmax`` with a default temperature of 0.0
        :param post_func: a function to post-process messages, same as ``post_process_messages_func``
        :param config: a module to pass configurations, defaults to ``Config()``
        :param next_cache: customized method for next cache
        """
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_func if pre_func else pre_embedding_func
        self.embedding_func = embedding_func
        self.data_manager: DataManager = data_manager
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_func if post_func else post_process_messages_func
        self.config = config
        self.next_cache = next_cache

        _pin_faiss_threads()

        if getattr(config, "exact_match_enabled", True):
            self.exact_match_cache = ExactMatchCache(
                max_size=getattr(config, "exact_match_max_size", 10_000),
                ttl_seconds=getattr(config, "exact_match_ttl_seconds", 300.0),
            )
        else:
            self.exact_match_cache = None

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:  # pylint: disable=W0703
                if not os.getenv("IS_CI"):
                    gptcache_log.error(e)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Any],
        session_ids: Optional[List[Optional[str]]] = None,
        batch_size: int = 1,
    ) -> None:
        """Import data to GPTCache

        :param questions: preprocessed question Data
        :param answers: list of answers to questions
        :param session_ids: list of the session id.
        :param batch_size: number of questions to embed in one call.
            Values >1 pass a list to ``embedding_func`` and expect a 2-D
            array back (shape ``[batch_size, dim]``), which is the case for
            all embedders that accept list input (e.g. ``SBERTMRL``).
            Defaults to 1 (original one-at-a-time behaviour).
        :type batch_size: int
        :return: None
        """
        if batch_size > 1:
            embedding_datas = []
            for i in range(0, len(questions), batch_size):
                batch = questions[i : i + batch_size]
                result = self.embedding_func(batch)
                # batch call returns (N, dim); single call returns (dim,)
                if hasattr(result, "ndim") and result.ndim == 2:
                    embedding_datas.extend(result)
                else:
                    embedding_datas.append(result)
        else:
            embedding_datas = [self.embedding_func(question) for question in questions]

        self.data_manager.import_data(
            questions=questions,
            answers=answers,
            embedding_datas=embedding_datas,
            session_ids=session_ids if session_ids else [None for _ in range(len(questions))],
        )

    def flush(self):
        """Flush data, to prevent accidental loss of memory data,
        such as using map cache management or faiss, hnswlib vector storage will be useful
        """
        self.data_manager.flush()
        if self.next_cache:
            self.next_cache.data_manager.flush()

    @staticmethod
    def set_openai_key():
        import_openai()
        import openai  # pylint: disable=C0415

        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def set_azure_openai_key():
        import_openai()
        import openai  # pylint: disable=C0415

        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")

    @staticmethod
    def set_anthropic_key():
        import_anthropic()
        import anthropic  # pylint: disable=C0415

        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

cache = Cache()
