import setuptools
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
    name="gpt_cache",
    packages=['gpt_cache'],
    version="0.0.1",
    author="SimFG",
    author_email="1142838399@qq.com",
    description="GPT Cache, make your chatgpt services lower cost and faster",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=parse_requirements('requirements.txt'),
    url="https://github.com/SimFG/gpt-cache",
    license='http://www.apache.org/licenses/LICENSE-2.0',
    python_requires='>=3.8.8',
)