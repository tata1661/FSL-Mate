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

import paddle
from paddlefsl.backbones import LinearBlock, MLP


def linear_block_test():
    train_input = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]])
    mlp = paddle.nn.Sequential(
        LinearBlock(2, 5),
        LinearBlock(5, 1)
    )
    print(mlp(train_input))  # Tensor of shape [2, 1]


def mlp_test():
    train_input = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]])
    mlp = MLP(input_size=2, output_size=1, hidden_sizes=[5])
    print(mlp(train_input))  # Tensor of shape [2, 1]
    for p in mlp.parameters():
        print(p)  # Parameter Tensor


if __name__ == '__main__':
    linear_block_test()
    mlp_test()
