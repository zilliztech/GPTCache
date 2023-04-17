import logging

FORMAT = '%(asctime)s - %(thread)d - %(filename)s-%(module)s:%(lineno)s - %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)

gptcache_log = logging.getLogger('gptcache')
