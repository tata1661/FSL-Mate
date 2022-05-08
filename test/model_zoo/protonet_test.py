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
from paddlefsl.model_zoo import protonet


def get_prototype_test():
    support_embeddings = paddle.to_tensor([[1.1, 1.1, 1.1],
                                           [0.0, 0.0, 0.0],
                                           [0.9, 0.9, 0.9],
                                           [0.0, 0.0, 0.0]])
    support_labels = paddle.to_tensor([[1], [0], [1], [0]])
    prototypes = protonet.get_prototypes(support_embeddings, support_labels, ways=2, shots=2)
    print(prototypes)  # Tensor of [[0, 0, 0], [1, 1, 1]]


if __name__ == '__main__':
    get_prototype_test()
