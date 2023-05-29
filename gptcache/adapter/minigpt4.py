from argparse import Namespace

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
# pylint: disable=wildcard-import
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import DataType, Question, Answer
from gptcache.utils.error import CacheError


class MiniGPT4:  # pragma: no cover
    """MiniGPT4 Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_image_question
            from gptcache.adapter.minigpt4 import MiniGPT4

            # init gptcache
            cache.init(pre_embedding_func=get_image_question)

            # run with gptcache
            pipe = MiniGPT4.from_pretrained(cfg_path='eval_configs/minigpt4_eval.yaml', gpu_id=3, options=None)
            question = "Which city is this photo taken?"
            image = "./merlion.png"
            answer = pipe(image, question)
    """
    def __init__(self, chat, return_hit):
        self.chat = chat
        self.return_hit = return_hit

    @classmethod
    def from_pretrained(cls, cfg_path, gpu_id=0, options=None,  return_hit=False):
        args = Namespace(cfg_path=cfg_path, gpu_id=gpu_id, options=options)
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
        return cls(chat, return_hit)

    def _llm_handler(self, image, question):
        chat_state = CONV_VISION.copy()
        img_list = []
        try:
            self.chat.upload_img(image, chat_state, img_list)
            self.chat.ask(question, chat_state)
            answer = self.chat.answer(conv=chat_state, img_list=img_list)[0]
            return answer if not self.return_hit else answer, False
        except Exception as e:
            raise CacheError("minigpt4 error") from e

    def __call__(self, image, question, *args, **kwargs):
        cache_context = {"deps": [
            {"name": "text", "data": question, "dep_type": DataType.STR},
            {"name": "image", "data": image, "dep_type": DataType.STR},
        ]}

        def cache_data_convert(cache_data):
            return cache_data if not self.return_hit else cache_data, True

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            question_data = Question.from_dict({
                "content": "pre_embedding_data",
                "deps": [
                    {"name": "text", "data": kwargs["question"], "dep_type": DataType.STR},
                    {"name": "image", "data": kwargs["image"], "dep_type": DataType.STR},
                ]
            })
            llm_data_cache = llm_data if not self.return_hit else llm_data[0]
            update_cache_func(Answer(llm_data_cache, DataType.STR), question=question_data)
            return llm_data

        return adapt(
            self._llm_handler, cache_data_convert, update_cache_callback, image=image, question=question, cache_context=cache_context, *args, **kwargs
        )
