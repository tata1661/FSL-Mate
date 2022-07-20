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
from __future__ import annotations

import importlib


def is_valid_module(module_name: str) -> bool:
    """checking that if the module has been installed

    Args:
        module_name (str): the name of the module

    Returns:
        bool: if the module has been installed
    """
   return importlib.util.find_spec(module_name) is not None 