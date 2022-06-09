# Copyright 2021 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import setuptools


def read_long_description() -> str:
    """read long description from README.md"""
    with open('README.md', encoding='utf-8') as readme_file:
        description = readme_file.read()
    return description


def read_version() -> str:
    """read version from VERSION"""
    with open('VERSION', encoding='utf-8') as version_file:
        version = version_file.read().strip()
    return version


def read_require_packages() -> List[str]:
    """read_require_packages from requirements.txt"""
    with open('requirements.txt', encoding='utf-8') as req_file:
        req_packages: List[str] = req_file.read().splitlines()
    return req_packages


setuptools.setup(
    name="paddlefsl",
    version=read_version(),
    author="tata1661 <YaQing Wang>",
    author_email="wangyaqing01@baidu.com",
    description="PaddleFSL is a Python library for few-shot learning",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL",
    packages=setuptools.find_packages(
        where='PaddleFSL',
        exclude=('examples',)
    ),
    package_dir={'': 'PaddleFSL'},
    setup_requires=['cython'],
    install_requires=read_require_packages(),
    ext_package='examples',
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0',
    keywords=["few-shot-learning", "one-shot-learning", "meta-learning", "paddlepaddle","deep-learning"]
)
