# Copyright 2022 PaddleFSL Authors
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

import paddle
from paddlefsl.backbones import distLinear


def distLinear_test():
    train_input = paddle.ones(shape=(25, 1024), dtype='float32')  # 5: 5 ways
    classifier = distLinear(indim=1024, outdim=64)
    print(classifier(train_input))  # Tensor of shape [25, 64]


if __name__ == '__main__':
    distLinear_test()

