import time

from gptcache import cache


def time_cal(func, func_name=None, report_func=None):
    def inner(*args, **kwargs):
        time_start = time.time()
        res = func(*args, **kwargs)
        delta_time = time.time() - time_start
        if cache.config.log_time_func:
            cache.config.log_time_func(
                func.__name__ if func_name is None else func_name, delta_time
            )
        if report_func is not None:
            report_func(delta_time)
        return res

    return inner
