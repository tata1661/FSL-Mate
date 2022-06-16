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
from paddlefsl.backbones import gnn_iclr


def gnn_iclr_test():
    N = 5  # 5: 5 ways
    K = 1  # 1: 1 shots
    hidden_size = 230
    model = gnn_iclr.GNN(N, hidden_size)
    model.eval()
    x_support = paddle.randn([1, 5, hidden_size])
    x_query = paddle.randn([1, 80, hidden_size])
    output = model(x_support, x_query, N, K, N * 16)
    print(output.shape)
    print(output)  # Tensor of shape [80, 5]


if __name__ == '__main__':
    gnn_iclr_test()
