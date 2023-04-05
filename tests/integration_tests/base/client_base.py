import os
import pytest
from utils.util_log import test_log as log
from common import common_type as ct
from common import common_func as cf


class Base:

    def setup_method(self, method):
        log.info(("*" * 35) + " setup " + ("*" * 35))
        log.info("[setup_method] Start setup test case %s." % method.__name__)
        log.info("[setup_method] Clean up tmp files.")
        cf.remove_file()

    def teardown_method(self, method):
        log.info(("*" * 35) + " teardown " + ("*" * 35))
        log.info("[teardown_method] Start teardown test case %s..." % method.__name__)
        log.info("[teardown_method] Clean up tmp files.")
        cf.remove_file()



