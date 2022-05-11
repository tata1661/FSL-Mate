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
from paddlefsl.model_zoo import maml


def inner_adapt_test():
    img1, label1 = paddle.ones(shape=(1, 1, 2, 2), dtype='float32'), paddle.to_tensor([[0]], dtype='int64')
    img2, label2 = paddle.zeros(shape=(1, 1, 2, 2), dtype='float32'), paddle.to_tensor([[1]], dtype='int64')
    model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(4, 2)
    )
    loss_fn = paddle.nn.CrossEntropyLoss()
    data = ((img1, label1), (img2, label2))
    maml.inner_adapt(model, data, loss_fn, 0.4)


if __name__ == '__main__':
    inner_adapt_test()
