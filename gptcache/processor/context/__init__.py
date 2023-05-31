from gptcache.utils.lazy_import import LazyImport

summarization = LazyImport(
    "summarization_context",
    globals(),
    "gptcache.processor.context.summarization_context",
)
selective = LazyImport(
    "selective_context", globals(), "gptcache.processor.context.selective_context"
)
concat = LazyImport(
    "concat_context", globals(), "gptcache.processor.context.concat_context"
)


__all__ = [
    "SummarizationContextProcess",
    "SelectiveContextProcess",
    "ConcatContextProcess",
]


def SummarizationContextProcess(model_name=None, tokenizer=None, target_length=512):
    return summarization.SummarizationContextProcess(
        model_name, tokenizer, target_length
    )


def SelectiveContextProcess(
    model_type: str = "gpt2",
    lang: str = "en",
    reduce_ratio: float = 0.35,
    reduce_level: str = "phrase",
):
    return selective.SelectiveContextProcess(
        model_type=model_type,
        lang=lang,
        reduce_ratio=reduce_ratio,
        reduce_level=reduce_level,
    )


def ConcatContextProcess():
    return concat.ConcatContextProcess()
