import codecs
import os
import re
from typing import List

import setuptools
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))


with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="gptcache",
    packages=find_packages(),
    version=find_version("gptcache", "__init__.py"),
    author="SimFG",
    author_email="bang.fu@zilliz.com",
    description="GPTCache, a powerful caching library that can be used to speed up and lower the cost of chat "
                "applications that rely on the LLM service. GPTCache works as a memcache for AIGC applications, "
                "similar to how Redis works for traditional applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=parse_requirements('requirements.txt'),
    url="https://github.com/zilliztech/GPTCache",
    license='https://opensource.org/license/mit/',
    python_requires='>=3.8.1',
    entry_points={
        'console_scripts': [
            'gptcache_server=gptcache_server.server:main',
        ],
    },
)
