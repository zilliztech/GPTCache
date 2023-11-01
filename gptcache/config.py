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
