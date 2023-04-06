import setuptools
from setuptools import find_packages
from typing import List

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]


setuptools.setup(
    name="gptcache",
    packages=find_packages(),
    version="0.1.3",
    author="SimFG",
    author_email="bang.fu@zilliz.com",
    description="GPT Cache, a powerful caching library that can be used to speed up and lower the cost of chat "
                "applications that rely on the LLM service. GPT Cache works as a memcache for AIGC applications, "
                "similar to how Redis works for traditional applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=parse_requirements('requirements.txt'),
    url="https://github.com/zilliztech/gptcache",
    license='http://www.apache.org/licenses/LICENSE-2.0',
    python_requires='>=3.8.1',
)