from typing import Dict, Any

import numpy as np

from gptcache.processor import ContextProcess
from gptcache.utils import import_huggingface

import_huggingface()

import transformers  # pylint: disable=C0413


def summarize_to_length(summarizer, text, target_len, max_len=1024):
    tokenizer = summarizer.tokenizer

    def token_length(text):
        return len(tokenizer.encode(text))

    segment_len = max_len - 100
    summary_result = text
    while token_length(text) > target_len:
        tokens = tokenizer.encode(text)
        segments = [
            tokens[i : i + segment_len] for i in range(0, len(tokens), segment_len - 1)
        ]
        summary_result = ""
        for segment in segments:
            len_seg = int(len(segment) / 4)
            summary = summarizer(
                tokenizer.decode(segment),
                min_length=max(len_seg - 10, 1),
                max_length=len_seg,
            )
            summary_result += summary[0]["summary_text"]
        text = summary_result
    return summary_result


class SummarizationContextProcess(ContextProcess):
    """A context processor for summarizing large amounts of text data using a summarizer model.

    :param summarizer: The summarizer model to use for summarization.
    :type summarizer: transformers.PreTrainedModel
    :param tokenizer: The tokenizer to use for tokenizing the text data.
    It used for measuring the output length.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param target_length: The length of the summarized text.
    :type target_length: int

    Example:
        .. code-block:: python

            from gptcache.processor.context.summarization_context import SummarizationContextProcess

            context_process = SummarizationContextProcess()
            cache.init(pre_embedding_func=context_process.pre_process)
    """

    def __init__(
        self, model_name="facebook/bart-large-cnn", tokenizer=None, target_length=512
    ):
        summarizer = transformers.pipeline(task="summarization", model=model_name)
        self.summarizer = summarizer
        self.target_length = target_length
        if tokenizer is None:
            tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        self.tokenizer = tokenizer
        self.content = ""

    def summarize_to_sentence(self, sentences, target_size=1000):
        lengths = []
        for sentence in sentences:
            lengths.append(len(sentence))
        total_length = np.array(lengths).sum()
        target_lengths = [int(target_size * l / total_length) for l in lengths]
        target_sentences = []
        for sent, target_len in zip(sentences, target_lengths):
            if len(self.tokenizer.tokenize(sent)) > target_len:
                response = summarize_to_length(
                    self.summarizer, sent, target_len, self.tokenizer.model_max_length
                )
                target_sentence = response
            else:
                target_sentence = sent
            target_sentences.append(target_sentence)
        result = ""
        for target_sentence in target_sentences:
            result = result + target_sentence
        return result

    def format_all_content(self, data: Dict[str, Any], **params: Dict[str, Any]):
        contents = []
        for query in data["messages"]:
            contents.append(query)
        self.content = contents

    def process_all_content(self) -> (Any, Any):
        def serialize_content(content):
            ret = ""
            for message in content:
                ret += "[#RS]{}[#RE][#CS]{}[#CE]".format(
                    message["role"], message["content"]
                )
            return ret

        result = self.summarize_to_sentence(
            [message["content"] for message in self.content], self.target_length
        )
        save_content = serialize_content(self.content)
        embedding_content = result
        return save_content, embedding_content
