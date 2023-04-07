"""" Methods of processing data """
import os
from common import common_type as ct
from utils.util_log import test_log as log


def remove_file(file_names=[ct.sqlite_file, ct.faiss_file]):
    """
    delete files
    :param file_names: file name list
    :return: None
    """
    for file in file_names:
        if os.path.isfile(file):
            os.remove(file)
            log.info("%s is removed" % file)


def log_time_func(func_name, delta_time):
    """
    print function time
    :param func_name: function name
    :param delta_time: consumed time
    :return: None
    """
    log.info("func `{}` consume time: {:.2f}s".format(func_name, delta_time))


def disable_cache(*args, **kwargs):
    """
    disable cache
    """
    return False
