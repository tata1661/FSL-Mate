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

import os
import setuptools
import sys

with open("README.md", "r") as fin:
    long_description = fin.read()

setuptools.setup(
    name="paddlefsl",
    version='1.1.0',
    author="PaddleFSL authors",
    author_email="wangyaqing01@baidu.com",
    description="PaddleFSL is a Python library for few-shot learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL",
    packages=setuptools.find_packages(
        where='.', exclude=('examples*', 'tests*')),
    setup_requires=['cython'],
    install_requires=[
    'numpy',
    'requests',
    'tqdm'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0',
    keywords=["few-shot-learning", "one-shot-learning", "meta-learning", "paddlepaddle","deep-learning"])
