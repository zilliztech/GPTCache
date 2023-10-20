from typing import Optional, List, Any, Mapping

from gptcache.adapter.adapter import adapt, aadapt
from gptcache.core import cache
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.session import Session
from gptcache.utils import import_langchain

import_langchain()

# pylint: disable=C0413
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    BaseMessage,
    LLMResult,
    AIMessage,
    ChatGeneration,
    ChatResult,
)
from langchain.callbacks.manager import (
    Callbacks,
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun,
)


# pylint: disable=protected-access
class LangChainLLMs(LLM):
    """LangChain LLM Wrapper.

    :param llm: LLM from langchain.llms.
    :type llm: Any

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from langchain.llms import OpenAI
            from gptcache.adapter.langchain_models import LangChainLLMs
            # run llm with gptcache
            llm = LangChainLLMs(llm=OpenAI(temperature=0))
            llm("Hello world")
    """

    llm: Any
    session: Session = None
    tmp_args: Any = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self.llm._identifying_params

    def __str__(self) -> str:
        return str(self.llm)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        _: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        session = (
            self.session
            if "session" not in self.tmp_args
            else self.tmp_args.pop("session")
        )
        cache_obj = self.tmp_args.pop("cache_obj", cache)
        return adapt(
            self.llm,
            _cache_data_convert,
            _update_cache_callback,
            prompt=prompt,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            **self.tmp_args,
        )

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None,
                     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None) -> str:
        return await super()._acall(prompt, stop=stop, run_manager=run_manager)

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return super().generate(prompts, stop=stop, callbacks=callbacks)

    async def agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks)

    def __call__(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        return (
            self.generate([prompt], stop=stop, callbacks=callbacks, **kwargs)
            .generations[0][0]
            .text
        )


# pylint: disable=protected-access
class LangChainChat(BaseChatModel):
    """LangChain LLM Wrapper.

    :param chat: LLM from langchain.chat_models.
    :type chat: Any

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_messages_last_content
            # init gptcache
            cache.init(pre_embedding_func=get_messages_last_content)
            cache.set_openai_key()
            from langchain.chat_models import ChatOpenAI
            from gptcache.adapter.langchain_models import LangChainChat
            # run chat model with gptcache
            chat = LangChainChat(chat=ChatOpenAI(temperature=0))
            chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    """

    @property
    def _llm_type(self) -> str:
        return "gptcache_llm_chat"

    chat: Any
    session: Optional[Session] = None
    tmp_args: Optional[Any] = None

    def _generate(
        self,
        messages: Any,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        session = (
            self.session
            if "session" not in self.tmp_args
            else self.tmp_args.pop("session")
        )
        cache_obj = self.tmp_args.pop("cache_obj", cache)
        return adapt(
            self.chat._generate,
            _cache_msg_data_convert,
            _update_cache_msg_callback,
            messages=messages,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            run_manager=run_manager,
            **self.tmp_args,
        )

    async def _agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        session = (
            self.session
            if "session" not in self.tmp_args
            else self.tmp_args.pop("session")
        )
        cache_obj = self.tmp_args.pop("cache_obj", cache)
        return await aadapt(
            self.chat._agenerate,
            _cache_msg_data_convert,
            _update_cache_msg_callback,
            messages=messages,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            run_manager=run_manager,
            **self.tmp_args,
        )

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return super().generate(messages, stop=stop, callbacks=callbacks)

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return await super().agenerate(messages, stop=stop, callbacks=callbacks)

    @property
    def _identifying_params(self):
        return self.chat._identifying_params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return self.chat._combine_llm_outputs(llm_outputs)

    def get_num_tokens(self, text: str) -> int:
        return self.chat.get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        return self.chat.get_num_tokens_from_messages(messages)

    def __call__(self, messages: Any, stop: Optional[List[str]] = None, **kwargs):
        generation = self.generate([messages], stop=stop, **kwargs).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")


def _cache_data_convert(cache_data):
    return cache_data


def _update_cache_callback(
    llm_data, update_cache_func, *args, **kwargs
):  # pylint: disable=unused-argument
    update_cache_func(Answer(llm_data, DataType.STR))
    return llm_data


def _cache_msg_data_convert(cache_data):
    llm_res = ChatResult(
        generations=[
            ChatGeneration(
                text="",
                generation_info=None,
                message=AIMessage(content=cache_data, additional_kwargs={}),
            )
        ],
        llm_output=None,
    )
    return llm_res


def _update_cache_msg_callback(
    llm_data, update_cache_func, *args, **kwargs
):  # pylint: disable=unused-argument
    update_cache_func(llm_data.generations[0].text)
    return llm_data
