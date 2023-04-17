from gptcache.utils.log import gptcache_log


def test_error_type():
    gptcache_log.setLevel("INFO")
    gptcache_log.error("Cache log error.")
    gptcache_log.warning("Cache log warning.")
    gptcache_log.info("Cache log info.")
    assert gptcache_log.level == 20
