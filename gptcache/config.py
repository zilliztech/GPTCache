from typing import Optional, Callable, List

from gptcache.utils.error import CacheError


class Config:
    """Pass configuration.

    :param log_time_func: optional, customized log time function
    :type log_time_func: Optional[Callable[[str, float], None]]
    :param similarity_threshold: a threshold ranged from 0 to 1 to filter search results with similarity score higher \
     than the threshold. When it is 0, there is no hits. When it is 1, all search results will be returned as hits.
    :type similarity_threshold: float
    :param prompts: optional, if the request content will remove the prompt string when the request contains the prompt list
    :type prompts: Optional[List[str]]
    :param template: optional, if the request content will remove the template string and only keep the parameter value in the template
    :type template: Optional[str]
    :param auto_flush: it will be automatically flushed every time xx pieces of data are added, default to 20
    :type auto_flush: int
    :param enable_token_counter: enable token counter, default to False
    :type enable_token_counter: bool
    :param input_summary_len: optional, summarize input to specified length.
    :type input_summary_len: Optional[int]
    :param skip_list: for sequence preprocessing, skip those sentences in skip_list.
    :type skip_list: Optional[List[str]]
    :param context_len: optional, the length of context.
    :type context_len: Optional[int]

    :param exact_match_enabled: enable the pre-embedding exact-match shortcut
        that hashes the normalized query and skips both the embedder and the
        vector search on an exact repeat. Defaults to True.
    :type exact_match_enabled: bool
    :param exact_match_max_size: max number of entries in the exact-match LRU.
        Defaults to 10000.
    :type exact_match_max_size: int
    :param exact_match_ttl_seconds: max age of an exact-match entry, in seconds.
        Bounds staleness when the semantic layer evicts an answer the exact-match
        cache still holds. Defaults to 300 seconds (5 minutes). Set to None to
        disable TTL (not recommended unless full coupling is wired up).
    :type exact_match_ttl_seconds: Optional[float]

    :param model_tier: pricing tier multiplier for the LLM being cached. Used by
        the CA_W_TINYLFU eviction policy to weight regeneration cost; a GPT-4-class
        model at ~$0.06/1k tokens might use ``model_tier=20`` relative to a
        GPT-3.5-class default of ``1.0``. Has no effect with other eviction policies.
    :type model_tier: float

    Example:
        .. code-block:: python

            from gptcache import Config

            configs = Config(similarity_threshold=0.6)
    """

    def __init__(
            self,
            log_time_func: Optional[Callable[[str, float], None]] = None,
            similarity_threshold: float = 0.8,
            prompts: Optional[List[str]] = None,
            template: Optional[str] = None,
            auto_flush: int = 20,
            enable_token_counter: bool = True,
            input_summary_len: Optional[int] = None,
            context_len: Optional[int] = None,
            skip_list: List[str] = None,
            data_check: bool = False,
            disable_report: bool = False,
            exact_match_enabled: bool = True,
            exact_match_max_size: int = 10000,
            exact_match_ttl_seconds: Optional[float] = 300.0,
            model_tier: float = 1.0,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError(
                "Invalid the similarity threshold param, reasonable range: 0-1"
            )
        self.log_time_func = log_time_func
        self.similarity_threshold = similarity_threshold
        self.prompts = prompts
        self.template = template
        self.auto_flush = auto_flush
        self.enable_token_counter = enable_token_counter
        self.input_summary_len = input_summary_len
        self.context_len = context_len
        if skip_list is None:
            skip_list = ["system", "assistant"]
        self.skip_list = skip_list
        self.data_check = data_check
        self.disable_report = disable_report
        self.exact_match_enabled = exact_match_enabled
        self.exact_match_max_size = exact_match_max_size
        self.exact_match_ttl_seconds = exact_match_ttl_seconds
        self.model_tier = model_tier
