# pylint: disable=unused-import, ungrouped-imports
try:
    from ruamel import yaml
except ModuleNotFoundError as moduleNotFound:
    try:
        from gptcache.utils.dependency_control import prompt_install
        prompt_install('ruamel.yaml')
        from ruamel import yaml
    except:
        from gptcache.utils.log import gptcache_log
        gptcache_log.error('ruamel.yaml not found, you can install via `pip install ruamel.yaml`.')
        raise ModuleNotFoundError('ruamel.yaml not found, you can install via `pip install ruamel.yaml`.') from moduleNotFound
